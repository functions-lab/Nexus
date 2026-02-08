/**
 * @file agora_worker.h
 * @brief Declaration file for the main Agora worker class
 */

#ifndef AGORA_WORKERSC_H_
#define AGORA_WORKERSC_H_

#include <memory>
#include <thread>
#include <vector>

#include "agora_buffer.h"
#include "agora_bufferBase.h"
#include "agora_workerBase.h"
#include "config.h"
#include "csv_logger.h"
#include "doer.h"
#include "mac_scheduler.h"
#include "mat_logger.h"
#include "phy_stats.h"
#include "stats.h"

class AgoraWorkerSC : public WorkerBase {
 public:
  explicit AgoraWorkerSC(Config* cfg, MacScheduler* mac_sched, Stats* stats,
                         PhyStats* phy_stats, MessageInfo* message,
                         BufferBase* buffer, FrameInfo* frame);
  ~AgoraWorkerSC();

  void RunWorker();

 private:
  void InitializeWorker();

  std::vector<std::shared_ptr<Doer> > computers_vec;
  std::vector<EventType> events_vec;
  int tid;  // TODO: remove thread id for single-core
  size_t cur_qid;
  size_t empty_queue_itrs;
  bool empty_queue;

  const size_t base_worker_core_offset_;

  Config* const config_;

  MacScheduler* mac_sched_;
  Stats* stats_;
  PhyStats* phy_stats_;
  MessageInfo* message_;
  BufferBase* buffer_;
  FrameInfo* frame_;
};

#endif  // AGORA_WORKERSC_H_