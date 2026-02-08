/**
 * @file main.cc
 * @brief Main file for the agora server
 */
#include "agora.h"
#include "gflags/gflags.h"
#include "logger.h"
#include "signal_handler.h"
#include "version_config.h"

#include <gflags/gflags.h>
#include <immintrin.h>
#include <netinet/ether.h>
#include <rte_byteorder.h>
#include <rte_cycles.h>
#include <rte_debug.h>
#include <rte_distributor.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_flow.h>
#include <rte_ip.h>
#include <rte_malloc.h>
#include <rte_pause.h>
#include <rte_prefetch.h>
#include <rte_udp.h>
#include <unistd.h>

#include "agora.h"
#include "gflags/gflags.h"
#include "logger.h"
#include "phy_stats.h"
#include "rte_bbdev.h"
#include "rte_bbdev_op.h"
#include "rte_bus_vdev.h"
#include "rte_eal.h"
#include "rte_lcore.h"
#include "rte_malloc.h"
#include "rte_mbuf.h"
#include "rte_mempool.h"
#include "scrambler.h"
#include "signal_handler.h"
#include "version_config.h"
#define LCORE_ID_1 5

DEFINE_string(
    conf_file,
    TOSTRING(PROJECT_DIRECTORY) "/files/config/ci/tddconfig-sim-both.json",
    "Config filename");

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("conf_file : set the configuration filename");
  gflags::SetVersionString(GetAgoraProjectVersion());
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> conf_files;
  for (int i = 1; i < argc; i++) {
    conf_files.push_back(std::string(argv[i]));
    std::cout << "adding config file:  " << std::string(argv[i]) << std::endl;
  }

  std::string core_list =
      std::to_string(LCORE_ID_1);  // this is hard set to core 36
  core_list =
        core_list ;
  const char* rte_argv[] = {
      "txrx", "-l", core_list.c_str(), "-a", "cb:00.0", "-a", "cb:00.1", "-a",
      "cb:00.2", "-a", "cb:00.3", 
      "-a", "cb:00.4", "-a", "cb:00.5", "-a", "cb:00.6", "-a", "cb:00.7", 
      "-a", "cb:01.0", "-a", "cb:01.1", "-a", "cb:01.2", "-a", "cb:01.3", 
      "-a", "cb:01.4", "-a", "cb:01.5", "-a", "cb:01.6", "-a", "cb:01.7", 
      // "-a", "18:00.4", "-a", "18:00.5", "-a", "18:00.6", "-a", "18:00.7",
      // "-a", "99:00.0",
      // "-a", "18:01.0", "-a", "18:01.1", "-a", "18:01.2", "-a", "18:01.3", "-a",
      // "18:01.4", "-a", "18:01.5", "-a", "18:01.6", "-a", "18:01.7", 
      // "-a", "ca:00.0", "-a", "ca:00.1", "-a", "31:00.0", "-a", "31:00.1",
      "--log-level", "lib.eal:info", nullptr};
  int rte_argc = static_cast<int>(sizeof(rte_argv) / sizeof(rte_argv[0])) - 1;

  // Initialize DPDK environment
  int ret_check = rte_eal_init(rte_argc, const_cast<char**>(rte_argv));
  RtAssert(
      ret_check >= 0,
      "Failed to initialize DPDK.  Are you running with root permissions?");

  int nb_bbdevs = rte_bbdev_count();
  std::cout << "num bbdevs: " << nb_bbdevs << std::endl;
  if (nb_bbdevs == 0) rte_exit(EXIT_FAILURE, "No bbdevs detected!\n");

  AGORA_LOG_INIT();

  // For backwards compatibility
  std::cout << "argc is: " << argc << std::endl;
  std::cout << "check !!!!!!!!!!" << std::endl;

  std::vector<std::unique_ptr<Agora>> agora_instances;
  std::vector<std::unique_ptr<Config>> configs;
  std::vector<std::thread> threads;

  int ret;
  try {
    SignalHandler signal_handler;

    // Register signal handler to handle kill signal
    signal_handler.SetupSignalHandlers();

    for (const auto& conf_file : conf_files) {
      threads.emplace_back([&, conf_file]() {
        auto cfg = std::make_unique<Config>(conf_file.c_str());
        cfg->GenData();

        auto agora = std::make_unique<Agora>(cfg.get());

        // Protect shared resources with mutex if needed
        {
          static std::mutex mtx;
          std::lock_guard<std::mutex> lock(mtx);
          configs.push_back(std::move(cfg));
          agora_instances.push_back(std::move(agora));
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }

    threads.clear();  // Clear threads before starting new ones

    for (size_t i = 0; i < agora_instances.size(); i++) {
      threads.emplace_back([&, i]() { agora_instances[i]->Start(); });
    }

    // agora_cli->Start();
    ret = EXIT_SUCCESS;
  } catch (SignalException& e) {
    std::cerr << "SignalException: " << e.what() << std::endl;
    ret = EXIT_FAILURE;
  }

  for (auto& thread : threads) {
    thread.join();
  }

  PrintCoreAssignmentSummary();
  gflags::ShutDownCommandLineFlags();
  AGORA_LOG_SHUTDOWN();

  return ret;
}
