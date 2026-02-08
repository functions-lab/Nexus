/**
 * @file sender_cli.cc
 * @brief Main file for the sender executable
 */
#include <gflags/gflags.h>

#include "logger.h"
#include "sender.h"
#include "version_config.h"

DEFINE_uint64(num_threads, 4, "Number of sender threads");
DEFINE_uint64(core_offset, 0, "Core ID of the first sender thread");
DEFINE_uint64(frame_duration, 0, "Frame duration in microseconds");
DEFINE_uint64(inter_frame_delay, 0, "Delay between two frames in microseconds");
DEFINE_string(server_mac_addr, "ff:ff:ff:ff:ff:ff",
              "MAC address of the remote Agora server to send data to");
DEFINE_string(
    conf_file,
    TOSTRING(PROJECT_DIRECTORY) "/files/config/ci/tddconfig-sim-ul.json",
    "Config filename");
DEFINE_uint64(
    enable_slow_start, 1,
    "Send frames slower than the specified frame duration during warmup");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetVersionString(GetAgoraProjectVersion());
  // std::string filename = FLAGS_conf_file;
  // std::string filename_2 = "files/config/ci/tddconfig-sim-ul-fr2-mu3-100Mhz-cell2.json";
  std::vector<std::string> conf_files;
  std::cout << "argc is: " << argc << std::endl;
  for (int i = 1; i < argc; i++) {
    conf_files.push_back(std::string(argv[i]));
    std::cout << "adding config file:  " << std::string(argv[i]) << std::endl;
  }

  // std::cout <<"argv[2] is : " << std::string(argv[1]) << std::endl;
  std::vector<std::unique_ptr<Config>> configs;
  std::vector<std::unique_ptr<Sender>> sender_instances;
  std::vector<std::thread> sender_threads;

  std::cout << "check !!!!!!!!!!" << std::endl;

  AGORA_LOG_INIT();

  std::cout << "in sender_cli.cc" << std::endl;

  {
    for (const auto& conf_file : conf_files) {
      auto cfg = std::make_unique<Config>(conf_file.c_str());
      cfg->GenData();
      configs.push_back(std::move(cfg));
    }

    {
      std::cout << "FLAGS_num_threads is " << FLAGS_num_threads << std::endl;
      int i = 0;

      for (auto& cfg : configs) {
        sender_instances.push_back(std::make_unique<Sender>(
            cfg.get(), FLAGS_num_threads, FLAGS_core_offset + i,
            FLAGS_frame_duration, FLAGS_inter_frame_delay,
            FLAGS_enable_slow_start, FLAGS_server_mac_addr));
        i = i + 2;
      }

      for (size_t i = 0; i < sender_instances.size(); i++) {
        sender_threads.emplace_back(
            [&, i]() { sender_instances[i]->StartTx(); });
      }

      // Wait for all threads to complete
      for (auto& thread : sender_threads) {
        thread.join();
      }

    }  // end context sender
  }    // end context Config

  PrintCoreAssignmentSummary();
  gflags::ShutDownCommandLineFlags();
  AGORA_LOG_SHUTDOWN();
  return 0;
}
