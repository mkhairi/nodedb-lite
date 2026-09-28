//! Row push refusals waiting to go back to Origin.
//!
//! When this replica refuses a `RowPush`, the dispatch loop queues a
//! `RowPushReject` here. The push loop drains the queue each tick and sends
//! every refusal, so Origin records the row this replica does not hold.

use nodedb_types::error::{ErrorDetails, NodeDbError};
use nodedb_types::sync::wire::{RowPushMsg, RowPushRefusal, RowPushRejectMsg};

use super::state::SyncClient;

/// Most refusals held for the next tick. The queue drains every tick, so
/// the cap is reached only when refusals arrive faster than one tick sends
/// them. A refusal past the cap is logged and not sent.
pub(super) const ROW_PUSH_REJECT_CAP: usize = 1024;

impl SyncClient {
    /// Queue the refusal of `msg`, which failed to apply with `error`.
    pub async fn queue_row_push_reject(&self, msg: &RowPushMsg, error: &NodeDbError) {
        let reject = row_push_reject(msg, error);
        let mut queue = self.pending_row_push_rejects.lock().await;
        if queue.len() >= ROW_PUSH_REJECT_CAP {
            tracing::error!(
                collection = %reject.collection,
                document_id = %reject.document_id,
                sequence = reject.sequence,
                cap = ROW_PUSH_REJECT_CAP,
                "RowPushReject queue full; Origin is not told about this refused row"
            );
            return;
        }
        queue.push(reject);
    }

    /// Take every queued refusal.
    pub async fn drain_row_push_rejects(&self) -> Vec<RowPushRejectMsg> {
        std::mem::take(&mut *self.pending_row_push_rejects.lock().await)
    }

    /// Put back `unsent`, drained refusals a failed send did not deliver,
    /// ahead of any queued since. Refusals past the cap are logged and
    /// not sent.
    pub async fn requeue_row_push_rejects(&self, unsent: Vec<RowPushRejectMsg>) {
        let mut queue = self.pending_row_push_rejects.lock().await;
        let newer = std::mem::replace(&mut *queue, unsent);
        queue.extend(newer);
        if queue.len() > ROW_PUSH_REJECT_CAP {
            for dropped in queue.drain(ROW_PUSH_REJECT_CAP..) {
                tracing::error!(
                    collection = %dropped.collection,
                    document_id = %dropped.document_id,
                    sequence = dropped.sequence,
                    cap = ROW_PUSH_REJECT_CAP,
                    "RowPushReject queue full; Origin is not told about this refused row"
                );
            }
        }
    }
}

/// The refusal of `msg`. A payload that does not decode is `Malformed`.
/// Any other error is `ApplyFailed`.
fn row_push_reject(msg: &RowPushMsg, error: &NodeDbError) -> RowPushRejectMsg {
    let detail = error.to_string();
    let refusal = if matches!(error.details(), ErrorDetails::Serialization { .. }) {
        RowPushRefusal::Malformed { detail }
    } else {
        RowPushRefusal::ApplyFailed { detail }
    };
    RowPushRejectMsg {
        collection: msg.collection.clone(),
        document_id: msg.document_id.clone(),
        sequence: msg.sequence,
        peer_id: msg.peer_id,
        refusal,
    }
}

#[cfg(test)]
mod tests {
    use nodedb_types::sync::wire::RowOp;

    use super::*;
    use crate::sync::client::SyncConfig;

    fn row_push() -> RowPushMsg {
        RowPushMsg {
            collection: "cfg".into(),
            document_id: "k1".into(),
            payload: vec![0xc0],
            op: RowOp::Upsert,
            lsn: 9,
            peer_id: 4,
            sequence: 12,
        }
    }

    #[tokio::test]
    async fn a_decode_error_is_queued_as_malformed() {
        let client = SyncClient::new(SyncConfig::new("wss://localhost:9090/sync", "t"));
        let error = NodeDbError::serialization("msgpack", "not a row map");
        client.queue_row_push_reject(&row_push(), &error).await;

        let drained = client.drain_row_push_rejects().await;
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].collection, "cfg");
        assert_eq!(drained[0].document_id, "k1");
        assert_eq!(drained[0].sequence, 12);
        assert_eq!(drained[0].peer_id, 4);
        assert!(matches!(
            drained[0].refusal,
            RowPushRefusal::Malformed { .. }
        ));
        assert!(client.drain_row_push_rejects().await.is_empty());
    }

    #[tokio::test]
    async fn a_storage_error_is_queued_as_apply_failed() {
        let client = SyncClient::new(SyncConfig::new("wss://localhost:9090/sync", "t"));
        let error = NodeDbError::storage("disk full");
        client.queue_row_push_reject(&row_push(), &error).await;
        let drained = client.drain_row_push_rejects().await;
        assert!(matches!(
            drained[0].refusal,
            RowPushRefusal::ApplyFailed { .. }
        ));
    }

    #[tokio::test]
    async fn requeued_refusals_go_ahead_of_newer_ones() {
        let client = SyncClient::new(SyncConfig::new("wss://localhost:9090/sync", "t"));
        let error = NodeDbError::storage("disk full");
        let mut first = row_push();
        first.sequence = 1;
        client.queue_row_push_reject(&first, &error).await;
        let unsent = client.drain_row_push_rejects().await;

        let mut second = row_push();
        second.sequence = 2;
        client.queue_row_push_reject(&second, &error).await;
        client.requeue_row_push_rejects(unsent).await;

        let order: Vec<u64> = client
            .drain_row_push_rejects()
            .await
            .iter()
            .map(|r| r.sequence)
            .collect();
        assert_eq!(order, vec![1, 2]);
    }
}
