/**
 * @file test_ldpc_baseband.cc
 * @brief Test LDPC performance after encoding, modulation, demodulation,
 * and decoding when different levels of
 * Gaussian noise is added to CSI
 */
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
#include <time.h>
#include <unistd.h>

#include <bitset>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "armadillo"
#include "comms-lib.h"
#include "config.h"
#include "data.h"
#include "data_generator.h"
#include "datatype_conversion.h"
#include "gettime.h"
#include "memory_manage.h"
#include "modulation.h"
#include "phy_ldpc_decoder_5gnr.h"
#include "rte_bbdev.h"
#include "rte_bbdev_op.h"
#include "rte_bus_vdev.h"
#include "rte_eal.h"
#include "rte_lcore.h"
#include "rte_malloc.h"
#include "rte_mbuf.h"
#include "rte_mempool.h"
#include "utils_ldpc.h"

struct LDPCConfig {
  uint8_t basegraph;
  uint16_t z_c;
  uint16_t n_filler;
  uint8_t rv_index;
  uint16_t n_cb;
  uint8_t q_m;
  uint8_t code_block_mode;
  uint32_t e;
  uint8_t iter_max;
};

LDPCConfig configs[] = {
    // Configuration 0 (MCS10)
    {1, 44, 0, 0, 2904, 4, 1, 2904, 20},
    // Configuration 1 (MCS12)
    {1, 56, 0, 0, 2912, 4, 1, 2912, 20},
    // Configuration 2 (MCS14)
    {1, 72, 0, 0, 2952, 4, 1, 2952, 20},
    // Configuration 3 (MCS16)
    {1, 88, 0, 0, 2992, 4, 1, 2992, 20},
    // Configuration 4 (MCS17)
    {1, 88, 0, 0, 4488, 6, 1, 4488, 20},
    // Configuration 5 (MCS18)
    {1, 88, 0, 0, 4224, 6, 1, 4224, 20},
    // Configuration 6 (MCS20)
    {1, 112, 0, 0, 4480, 6, 1, 4480, 20},
    // Configuration 7 (MCS22)
    {1, 128, 0, 0, 4352, 6, 1, 4352, 20},
    // Configuration 8 (MCS24)
    {1, 144, 0, 0, 4176, 6, 1, 4176, 20},
    // Configuration 9 (MCS26)
    {1, 176, 0, 0, 4576, 6, 1, 4576, 20},
    // Configuration 10 (MCS28)
    {1, 192, 0, 0, 4608, 6, 1, 4608, 20}};

// scrambler_(std::make_unique<AgoraScrambler::Scrambler>());

int test_memcmp_avx2(const uint8_t* data1, const uint8_t* data2,
                     size_t num_bytes) {
  size_t num_blocks = num_bytes / 32;  // Process 32 bytes at a time for AVX2
  for (size_t i = 0; i < num_blocks; i++) {
    __m256i chunk1 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data1 + i * 32));
    __m256i chunk2 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data2 + i * 32));
    __m256i result =
        _mm256_xor_si256(chunk1, chunk2);       // XOR each byte, 0 if equal
    if (!_mm256_testz_si256(result, result)) {  // Check if any byte differs
      return -1;
    }
  }
  // Compare any remaining bytes
  size_t remainder = num_bytes % 32;
  for (size_t i = num_bytes - remainder; i < num_bytes; i++) {
    if (data1[i] != data2[i]) return -1;
  }
  return 0;
}

#define GET_SOCKET(socket_id) (((socket_id) == SOCKET_ID_ANY) ? 0 : (socket_id))
#define MAX_RX_BYTE_SIZE 1500
#define CACHE_SIZE 128
#define NUM_QUEUES 4
#define LCORE_ID 36
#define NUM_ELEMENTS_IN_POOL 2047
#define NUM_ELEMENTS_IN_MEMPOOL 16383
#define DATA_ROOM_SIZE 45488
#define CPU_FREQ_HZ 2600000000ULL  // 2.9 GHz, for example

#define NB_MBUF 8192
#define TEST_SUCCESS 0
#define TEST_FAILED -1
#define TEST_SKIPPED 1
#define MAX_QUEUES RTE_MAX_LCORE
#define OPS_CACHE_SIZE 256U
#define MAX_PKT_BURST 32
#define MAX_BURST 512U
#define SYNC_START 1

typedef struct {
  double time;
  int retry_count;
  size_t BLER;
} DataPoint;

uint8_t dev_id;
int ldpc_llr_decimals;
int ldpc_llr_size;
uint32_t ldpc_cap_flags;
uint16_t min_alignment;
uint16_t num_ops = 2047;
uint16_t burst_sz = 1;
size_t q_m;
size_t e;
uint16_t enq = 0;
uint16_t deq = 0;
uint16_t enq_2 = 0;
uint16_t deq_2 = 0;
size_t enq_index = 0;
// const size_t num_ul_syms = cfg_->Frame().NumULSyms();

struct rte_mempool* ops_mp;
struct rte_mempool* ops_mp_2;

struct rte_mempool* in_mbuf_pool;
struct rte_mempool* out_mbuf_pool;
struct rte_mempool* bbdev_op_pool;

// struct rte_mempool* in_mbuf_pool_2;
// struct rte_mempool* out_mbuf_pool_2;
// struct rte_mempool* bbdev_op_pool_2;

struct rte_mbuf* in_mbuf;
struct rte_mbuf* out_mbuf;

struct rte_bbdev_dec_op* ref_dec_op[64];
struct rte_bbdev_dec_op* ops_deq[64];

// struct rte_bbdev_dec_op* ref_dec_op_2[64];
// struct rte_bbdev_dec_op* ops_deq_2[64];

struct rte_bbdev_op_data** inputs;
struct rte_bbdev_op_data** hard_outputs;

static inline bool check_bit(uint32_t bitmap, uint32_t bitmask) {
  return bitmap & bitmask;
}

void get_timestamped_filename(char* filename, size_t size, const char* folder,
                              int int_1, int int_2, int int_3, int int_4,
                              int int_5) {
  // Construct the full file path using the provided integers
  snprintf(filename, size, "%s/%d-Task-Half-%d-%d-%d-%d.csv", folder, int_5,
           int_1, int_2, int_3, int_4);
}

void test_print_casted_uint8_hex(const uint8_t* array, size_t totalByteLength) {
  for (size_t i = 0; i < totalByteLength; i++) {
    // Cast each int8_t to uint8_t and print as two-digit hexadecimal
    printf("%02X ", (uint8_t)array[i]);
  }
  printf("\n");  // Add a newline for readability
}

void write_csv_header(FILE* file) {
  fprintf(file, "Iteration,Time,Retry_Count,BLER\n");
}

int test_allocate_buffers_on_socket(struct rte_bbdev_op_data** buffers,
                                    const int len, const int socket) {
  int i;
  *buffers = static_cast<struct rte_bbdev_op_data*>(
      rte_zmalloc_socket(NULL, len, 0, socket));
  if (*buffers == NULL) {
    printf("WARNING: Failed to allocate op_data on socket %d\n", socket);
    /* try to allocate memory on other detected sockets */
    for (i = 0; i < socket; i++) {
      *buffers = static_cast<struct rte_bbdev_op_data*>(
          rte_zmalloc_socket(NULL, len, 0, i));
      if (*buffers != NULL) break;
    }
  }
  return (*buffers == NULL) ? TEST_FAILED : TEST_SUCCESS;
}

enum BaseGraph { BG1 = 1, BG2 = 2 };

int main(int argc, char* argv[]) {
  if (argc !=
      6) {  // Check if there are exactly 3 arguments (program name + 2 integers)
    printf("Usage: %s <int1> <int2> <int3> <int4> <int5>\n", argv[0]);
    return 1;
  }

  int config_1 = atoi(argv[1]);
  int config_2 = atoi(argv[2]);
  int config_3 = atoi(argv[3]);
  int config_4 = atoi(argv[4]);
  int iter_num = atoi(argv[5]);

  // EAL initialization && ACC100 Card device detection and initialization
  std::string core_list = std::to_string(36);
  // + "," + std::to_string(35) + "," + std::to_string(36) + "," + std::to_string(37) + "," + std::to_string(38);
  const char* rte_argv[] = {"txrx",         "-l",      core_list.c_str(),
                            "-a",           "18:00.1", "--log-level",
                            "lib.eal:info", nullptr};
  int rte_argc = static_cast<int>(sizeof(rte_argv) / sizeof(rte_argv[0])) - 1;

  // Initialize DPDK environment
  int ret = rte_eal_init(rte_argc, const_cast<char**>(rte_argv));
  RtAssert(
      ret >= 0,
      "Failed to initialize DPDK.  Are you running with root permissions?");

  int nb_bbdevs = rte_bbdev_count();
  std::cout << "num bbdevs: " << nb_bbdevs << std::endl;

  if (nb_bbdevs == 0) rte_exit(EXIT_FAILURE, "No bbdevs detected!\n");
  dev_id = 0;
  struct rte_bbdev_info info;
  // rte_bbdev_info_get(dev_id, &info);
  rte_bbdev_intr_enable(dev_id);
  rte_bbdev_info_get(dev_id, &info);

  bbdev_op_pool =
      rte_bbdev_op_pool_create("bbdev_op_pool_dec", RTE_BBDEV_OP_LDPC_DEC,
                               NB_MBUF, CACHE_SIZE, rte_socket_id());
  ret = rte_bbdev_setup_queues(dev_id, NUM_QUEUES, info.socket_id);

  if (ret < 0) {
    printf("rte_bbdev_setup_queues(%u, %u, %d) ret %i\n", dev_id, NUM_QUEUES,
           rte_socket_id(), ret);
  }

  ret = rte_bbdev_intr_enable(dev_id);

  struct rte_bbdev_queue_conf qconf;
  qconf.socket = info.socket_id;
  qconf.queue_size = info.drv.queue_size_lim;
  qconf.op_type = RTE_BBDEV_OP_LDPC_DEC;
  qconf.priority = 0;

  std::cout << "device id is: " << static_cast<int>(dev_id) << std::endl;

  for (int q_id = 0; q_id < NUM_QUEUES; q_id++) {
    /* Configure all queues belonging to this bbdev device */
    ret = rte_bbdev_queue_configure(dev_id, q_id, &qconf);
    if (ret < 0)
      rte_exit(EXIT_FAILURE,
               "ERROR(%d): BBDEV %u queue %u not configured properly\n", ret,
               dev_id, q_id);
  }

  ret = rte_bbdev_start(dev_id);
  int socket_id = GET_SOCKET(info.socket_id);

  in_mbuf_pool = rte_pktmbuf_pool_create("in_pool_0", NUM_ELEMENTS_IN_MEMPOOL,
                                         0, 0, DATA_ROOM_SIZE, 0);
  out_mbuf_pool = rte_pktmbuf_pool_create(
      "hard_out_pool_0", NUM_ELEMENTS_IN_MEMPOOL, 0, 0, DATA_ROOM_SIZE, 0);

  if (in_mbuf_pool == nullptr or out_mbuf_pool == nullptr) {
    std::cerr << "Error: Unable to create mbuf pool: "
              << rte_strerror(rte_errno) << std::endl;
  }

  // in_mbuf_pool_2 = rte_pktmbuf_pool_create("in_pool_1", NUM_ELEMENTS_IN_MEMPOOL, 0, 0, DATA_ROOM_SIZE, 0);
  // out_mbuf_pool_2 =
  //     rte_pktmbuf_pool_create("hard_out_pool_1", NUM_ELEMENTS_IN_MEMPOOL, 0, 0, DATA_ROOM_SIZE, 0);

  // if (in_mbuf_pool_2 == nullptr or out_mbuf_pool_2 == nullptr) {
  //   std::cerr << "Error: Unable to create mbuf pool: "
  //             << rte_strerror(rte_errno) << std::endl;
  // }

  ops_mp = rte_bbdev_op_pool_create("RTE_BBDEV_OP_LDPC_DEC_poo",
                                    RTE_BBDEV_OP_LDPC_DEC, NUM_ELEMENTS_IN_POOL,
                                    OPS_CACHE_SIZE, socket_id);
  if (ops_mp == nullptr) {
    std::cerr << "Error: Failed to create memory pool for bbdev operations."
              << std::endl;
  } else {
    std::cout << "Memory pool for bbdev operations created successfully."
              << std::endl;
  }

  // ops_mp_2 = rte_bbdev_op_pool_create("RTE_BBDEV_OP_LDPC_DEC_2",
  //                                   RTE_BBDEV_OP_LDPC_DEC, NUM_ELEMENTS_IN_POOL, OPS_CACHE_SIZE,
  //                                   socket_id);
  // if (ops_mp_2 == nullptr) {
  //   std::cerr << "Error: Failed to create memory pool for bbdev operations."
  //             << std::endl;
  // } else {
  //   std::cout << "Memory pool for bbdev operations created successfully."
  //             << std::endl;
  // }

  int rte_alloc_ref = rte_bbdev_dec_op_alloc_bulk(ops_mp, ref_dec_op, 16 * 4);
  if (rte_alloc_ref != TEST_SUCCESS) {
    rte_exit(EXIT_FAILURE, "Failed to alloc bulk\n");
  }

  // rte_alloc_ref = rte_bbdev_dec_op_alloc_bulk(ops_mp_2, ref_dec_op_2, 16 * 4);
  // if (rte_alloc_ref != TEST_SUCCESS ) {
  //   rte_exit(EXIT_FAILURE, "Failed to alloc bulk\n");
  // }

  const struct rte_bbdev_op_cap* cap = info.drv.capabilities;
  const struct rte_bbdev_op_cap* capabilities = NULL;
  rte_bbdev_info_get(dev_id, &info);

  for (unsigned int i = 0; cap->type != RTE_BBDEV_OP_NONE; ++i, ++cap) {
    std::cout << "cap is: " << cap->type << std::endl;
    if (cap->type == RTE_BBDEV_OP_LDPC_DEC) {
      capabilities = cap;
      std::cout << "capability is being set to: " << capabilities->type
                << std::endl;
      break;
    }
  }

  inputs =
      (struct rte_bbdev_op_data**)malloc(sizeof(struct rte_bbdev_op_data*));
  hard_outputs =
      (struct rte_bbdev_op_data**)malloc(sizeof(struct rte_bbdev_op_data*));

  std::cout << "555: here!!!!!!!!!!" << std::endl;

  int ret_socket_in = test_allocate_buffers_on_socket(
      inputs, 1 * sizeof(struct rte_bbdev_op_data), 0);
  int ret_socket_hard_out = test_allocate_buffers_on_socket(
      hard_outputs, 1 * sizeof(struct rte_bbdev_op_data), 0);

  if (ret_socket_in != TEST_SUCCESS || ret_socket_hard_out != TEST_SUCCESS) {
    rte_exit(EXIT_FAILURE, "Failed to allocate socket\n");
  }

  std::cout << "Init Successful" << std::endl;

  // Get the total number of elements in the values array
  size_t num_elements_10 = sizeof(values_10) / sizeof(values_10[0]);
  size_t ref_num_elements_10 = sizeof(ref_values_10) / sizeof(ref_values_10[0]);
  std::cout << "num_elements is: " << num_elements_10 << std::endl;
  uint8_t* int8_ptr_10 = reinterpret_cast<uint8_t*>(values_10);
  uint8_t* ref_int8_ptr_10 = reinterpret_cast<uint8_t*>(ref_values_10);
  size_t num_int8_elements_10 = num_elements_10 * 4;

  uint8_t* int8_ptr_12 = reinterpret_cast<uint8_t*>(values_12);
  uint8_t* ref_int8_ptr_12 = reinterpret_cast<uint8_t*>(ref_values_12);
  uint8_t* int8_ptr_14 = reinterpret_cast<uint8_t*>(values_14);
  uint8_t* ref_int8_ptr_14 = reinterpret_cast<uint8_t*>(ref_values_14);
  uint8_t* int8_ptr_16 = reinterpret_cast<uint8_t*>(values_16);
  uint8_t* ref_int8_ptr_16 = reinterpret_cast<uint8_t*>(ref_values_16);
  uint8_t* int8_ptr_17 = reinterpret_cast<uint8_t*>(values_17);
  uint8_t* ref_int8_ptr_17 = reinterpret_cast<uint8_t*>(ref_values_17);
  uint8_t* int8_ptr_18 = reinterpret_cast<uint8_t*>(values_18);
  uint8_t* ref_int8_ptr_18 = reinterpret_cast<uint8_t*>(ref_values_18);

  uint8_t* int8_ptr_20 = reinterpret_cast<uint8_t*>(values_20);
  uint8_t* ref_int8_ptr_20 = reinterpret_cast<uint8_t*>(ref_values_20);
  uint8_t* int8_ptr_22 = reinterpret_cast<uint8_t*>(values_22);
  uint8_t* ref_int8_ptr_22 = reinterpret_cast<uint8_t*>(ref_values_22);
  uint8_t* int8_ptr_24 = reinterpret_cast<uint8_t*>(values_24);
  uint8_t* ref_int8_ptr_24 = reinterpret_cast<uint8_t*>(ref_values_24);
  uint8_t* int8_ptr_26 = reinterpret_cast<uint8_t*>(values_26);
  uint8_t* ref_int8_ptr_26 = reinterpret_cast<uint8_t*>(ref_values_26);
  uint8_t* int8_ptr_28 = reinterpret_cast<uint8_t*>(values_28);
  uint8_t* ref_int8_ptr_28 = reinterpret_cast<uint8_t*>(ref_values_28);

  uint8_t* input_values_map[] = {int8_ptr_10, int8_ptr_12, int8_ptr_14,
                                 int8_ptr_16, int8_ptr_17, int8_ptr_18,
                                 int8_ptr_20, int8_ptr_22, int8_ptr_24,
                                 int8_ptr_26, int8_ptr_28};

  uint8_t* ref_values_map[] = {
      ref_int8_ptr_10, ref_int8_ptr_12, ref_int8_ptr_14, ref_int8_ptr_16,
      ref_int8_ptr_17, ref_int8_ptr_18, ref_int8_ptr_20, ref_int8_ptr_22,
      ref_int8_ptr_24, ref_int8_ptr_26, ref_int8_ptr_28};

  // int iter_num = 64;
  for (int i = 0; i < iter_num; i++) {
    LDPCConfig& config = (i < 16)   ? configs[config_1]
                         : (i < 32) ? configs[config_2]
                         : (i < 48) ? configs[config_3]
                                    : configs[config_4];

    ref_dec_op[i]->ldpc_dec.basegraph = config.basegraph;
    ref_dec_op[i]->ldpc_dec.z_c = config.z_c;
    ref_dec_op[i]->ldpc_dec.n_filler = config.n_filler;
    ref_dec_op[i]->ldpc_dec.rv_index = config.rv_index;
    ref_dec_op[i]->ldpc_dec.n_cb = config.n_cb;
    ref_dec_op[i]->ldpc_dec.q_m = config.q_m;
    ref_dec_op[i]->ldpc_dec.code_block_mode = config.code_block_mode;
    ref_dec_op[i]->ldpc_dec.cb_params.e = config.e;
    ref_dec_op[i]->ldpc_dec.iter_max = config.iter_max;
    ref_dec_op[i]->opaque_data = (void*)(uintptr_t)i;
  }

  std::cout << "init okay!!!" << std::endl;

  // Specify the folder name
  const char* folder = "../mcs_combine";
  char filename[150];
  get_timestamped_filename(filename, sizeof(filename), folder, config_1,
                           config_2, config_3, config_4, iter_num);

  struct rte_bbdev_op_ldpc_dec* ops_td;
  struct rte_bbdev_op_data* hard_output;
  struct rte_mbuf* temp_m;
  size_t memcmp_ret = 0;
  int test_iter_num = 100000;

  DataPoint* results = (DataPoint*)malloc(test_iter_num * sizeof(DataPoint));
  if (results == NULL) {
    printf("Memory allocation failed.\n");
    return 1;
  }

  for (int j = 0; j < test_iter_num; j++) {
    char* data;
    struct rte_bbdev_op_data* bufs = *inputs;
    rte_bbdev_op_data* bufs_out = *hard_outputs;
    struct rte_mbuf* m_head;
    struct rte_mbuf* m_head_out;

    size_t start = GetTime::WorkerRdtsc();
    for (int i = 0; i < iter_num; i++) {
      LDPCConfig& config = (i < 16)   ? configs[config_1]
                           : (i < 32) ? configs[config_2]
                           : (i < 48) ? configs[config_3]
                                      : configs[config_4];
      uint8_t* input_values = (i < 16)   ? input_values_map[config_1]
                              : (i < 32) ? input_values_map[config_2]
                              : (i < 48) ? input_values_map[config_3]
                                         : input_values_map[config_4];
      // std::cout << "i is: " << i << std::endl;
      m_head = rte_pktmbuf_alloc(in_mbuf_pool);

      bufs[0].data = m_head;
      bufs[0].offset = 0;
      bufs[0].length = 0;
      data = rte_pktmbuf_append(m_head, config.n_cb);

      // Copy data from demod_data to the mbuf
      rte_memcpy(data, input_values + (0 * config.n_cb), config.n_cb);
      bufs[0].length += config.n_cb;

      m_head_out = rte_pktmbuf_alloc(out_mbuf_pool);

      bufs_out[0].data = m_head_out;
      bufs_out[0].offset = 0;
      bufs_out[0].length = 0;

      bufs_out[0].length += config.n_cb;

      ref_dec_op[i]->ldpc_dec.input = *inputs[0];
      ref_dec_op[i]->ldpc_dec.hard_output = *hard_outputs[0];

      rte_pktmbuf_free(m_head);
      rte_pktmbuf_free(m_head_out);

      enq += rte_bbdev_enqueue_ldpc_dec_ops(0, 0, &ref_dec_op[i], 1);
    }

    // size_t start = GetTime::WorkerRdtsc();

    // while (enq < iter_num) {
    //   enq += rte_bbdev_enqueue_ldpc_dec_ops(0, 0, &ref_dec_op[enq], iter_num);
    // }

    // std::cout << "enq is: " << enq << std::endl;
    int retry_count = 0;
    while (deq < enq && retry_count < 100000000) {
      deq += rte_bbdev_dequeue_ldpc_dec_ops(0, 0, &ops_deq[deq], enq - deq);
      retry_count++;
    }

    size_t end = GetTime::WorkerRdtsc();
    double time = (double)(end - start) / CPU_FREQ_HZ * 1000.0;

    // ############################################################################################################
    // uint32_t len = 0;
    for (int i = 0; i < iter_num; i++) {
      ops_td = &ops_deq[i]->ldpc_dec;
      hard_output = &ops_td->hard_output;
      temp_m = hard_output->data;
      // len = hard_output->length;
      uint8_t* temp_data = rte_pktmbuf_mtod(temp_m, uint8_t*);

      LDPCConfig& config = (i < 16)   ? configs[config_1]
                           : (i < 32) ? configs[config_2]
                           : (i < 48) ? configs[config_3]
                                      : configs[config_4];
      uint8_t* ref_values = (i < 16)   ? ref_values_map[config_1]
                            : (i < 32) ? ref_values_map[config_2]
                            : (i < 48) ? ref_values_map[config_3]
                                       : ref_values_map[config_4];
      size_t memcmp_length = (config.z_c * 22) / 8;

      memcmp_ret = test_memcmp_avx2(reinterpret_cast<uint8_t*>(temp_data),
                                    ref_values, memcmp_length);
    }

    results[j].time = time;
    results[j].retry_count = retry_count;
    results[j].BLER = memcmp_ret;

    enq = 0;
    deq = 0;
    memcmp_ret = 0;
  }

  // Open the CSV file in write mode using the timestamped filename
  FILE* file = fopen(filename, "w");
  if (file == NULL) {
    printf("Could not open file for writing.\n");
    free(results);
    return 1;
  }

  // Write CSV header
  fprintf(file, "Iteration,Time,Retry_Count,BLER\n");

  // Write data from the vector to the file
  for (int j = 0; j < test_iter_num; j++) {
    fprintf(file, "%d,%f,%d,%ld\n", j, results[j].time, results[j].retry_count,
            results[j].BLER);
  }

  // Close the file and free allocated memory
  fclose(file);
  free(results);

  printf("Data has been written to %s\n", filename);

  return 0;
}
