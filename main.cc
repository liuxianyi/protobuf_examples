#include <iostream>
#include <deque>
#include <cmath>
#include <vector>
#include <string>

#include "build/message.pb.h"

class SlidingWindowStatistics {
 public:
  explicit SlidingWindowStatistics(size_t window_size) : window_size(window_size), sum(0.0) {}

  void addValue(double value) {
    if (window.size() < window_size) {
      window.push_back(value);
      sum += value;
    } else {
      // 移除最旧的值
      sum -= window.front();
      window.pop_front();
      // 添加新值
      window.push_back(value);
      sum += value;
    }
  }

  double calculateMean() const {
    if (window.empty()) {
      return 0.0;
    }
    return sum / window.size();
  }

  double calculateStdDev() const {
    if (window.empty()) {
      return 0.0;
    }
    double mean = calculateMean();
    double sum_squares = 0.0;

    for (const double& value : window) {
        sum_squares += (value - mean) * (value - mean);
    }

    return std::sqrt(sum_squares / window.size());
  }

  int size() const {
    return window.size();
  }

 private:
  size_t window_size = 0;
  std::deque<double> window = {};
  double sum = 0;
};
struct LowestCostRlInputData {
  std::vector<float> input_0_list;
  std::vector<int64_t> input_1_list;
  std::vector<float> input_2_list;
  std::vector<float> input_3_list;
  std::vector<int64_t> input_4_list;
  std::vector<int32_t> input_5_list;
};
class LowestCostRlFeatureTask {
public:
    LowestCostRlFeatureTask() = default;
    ~LowestCostRlFeatureTask() = default;
    static LowestCostRlFeatureTask* Instance() {
        static LowestCostRlFeatureTask instance;
        return &instance;
    }
    bool PushFeatures(NativeLowestCostContext* lowest_cost_context);
    bool ComputeFeatures(const NativeLowestCostContext& lowest_cost_context,
                       LowestCostRlInputData& lowest_cost_rl_input_data);   // NOLINT
};

bool LowestCostRlFeatureTask::PushFeatures(NativeLowestCostContext* p_bid_context) {
  std::vector<float> past_pv_ratio_list {-1.70538397e-08,  1.61504434e-03,  3.19944831e-03,  4.75294676e-03,
        6.32441437e-03,  7.89325925e-03,  1.02432836e-02,  1.17954707e-02,
        1.33401086e-02,  1.48917287e-02,  1.64433133e-02,  1.79907511e-02,
        1.95473330e-02,  2.18703151e-02,  2.34042495e-02,  2.49495962e-02,
        2.64794194e-02,  2.79847520e-02,  2.94915378e-02,  3.10111536e-02,
        3.32928455e-02,  3.48132411e-02,  3.63589068e-02,  3.79035447e-02};
  std::vector<float> past_cost_ratio_list {-1.45289223e-08,  4.99319033e-06,  2.21996053e-03,  4.21500553e-03,
        7.77496280e-03,  1.01349264e-02,  1.38199249e-02,  1.79298966e-02,
        2.13898892e-02,  2.55598398e-02,  2.83198518e-02,  3.07448399e-02,
        3.25048635e-02,  3.43698216e-02,  3.66798218e-02,  3.84848011e-02,
        4.00497893e-02,  4.16898173e-02,  4.34197946e-02,  4.41597610e-02,
        4.48647872e-02,  4.56597626e-02,  4.62497630e-02,  4.66347883e-02}; 
  std::vector<float> p_error_list { 0.01094113,  0.00214878,  0.00110892, -0.00275642, -0.0042148 ,
       -0.00664446, -0.01107381, -0.01434821, -0.01871288, -0.02081585,
       -0.02239467, -0.02289835, -0.02238599, -0.02379127, -0.02434849,
       -0.02452636, -0.02485973, -0.02533592, -0.02411006, -0.02150364,
       -0.02028858, -0.01863942, -0.01658865, -0.0150469 };
  std::vector<float> d_error_list {0.01094114, -0.00879235, -0.00103987, -0.00386538, -0.00145839,
       -0.00242958, -0.0044294 , -0.00327441, -0.00436469, -0.00210295,
       -0.00157883, -0.00050363,  0.00051236, -0.00140536, -0.00055721,
       -0.00017784, -0.00033334, -0.00047623,  0.00122585,  0.00260648,
        0.001215  ,  0.00164912,  0.00205082,  0.00154174};
  std::vector<float> pacing_rate_list {1.0711174 , 0.92218167, 0.9186205 , 0.8973569 , 0.8856393 ,
       0.86656797, 0.83303815, 0.80112123, 0.75911486, 0.725234  ,
       0.6920018 , 0.6632063 , 0.6397296 , 0.6058098 , 0.5752049 ,
       0.54618126, 0.516097  , 0.4848503 , 0.46284917, 0.45017204,
       0.4325255 , 0.41876182, 0.40918413, 0.39904585};
  std::vector<float> total_pred_cv_list = std::vector<float>(24, 1.0);
  for (int i = 0; i < 24; i++) {
      const auto& past_pv_ratio = past_pv_ratio_list[i];
    const auto& past_cost_ratio = past_cost_ratio_list[i];
    const auto& p_error = p_error_list[i];
    const auto& d_error = d_error_list[i];
    auto bid_ratio = pacing_rate_list[i];
    const auto& total_pred_cv = total_pred_cv_list[i];

    int32_t now_time_step = p_bid_context->mutable_rl_features()->now_time_step();
    
    auto* raw_features_ptr = p_bid_context->mutable_rl_features()->add_raw_features();
  // if (SPDM_enableAddPreCvFeat()) {
    raw_features_ptr->set_total_pred_cv(total_pred_cv);
    raw_features_ptr->set_past_pv_ratio(past_pv_ratio);
    raw_features_ptr->set_past_cost_ratio(past_cost_ratio);
    std::cout << "goog rlfeat: "
              << "past_pv_ratio: " << past_pv_ratio
              << "past_cost_ratio: " << past_cost_ratio
              << std::endl;
  // } else {
    // raw_features_ptr->set_past_pv_ratio(1 - past_pv_ratio);
    // raw_features_ptr->set_past_cost_ratio(1 - past_cost_ratio);
  // }
    raw_features_ptr->set_p_error(p_error);
    raw_features_ptr->set_d_error(d_error);
    raw_features_ptr->set_bid_ratio(bid_ratio);
    raw_features_ptr->set_time_step(now_time_step);
  if (now_time_step >= 1439) {
    now_time_step = -1;
  }
  p_bid_context->mutable_rl_features()->set_now_time_step(now_time_step + 1);
  }
  return true;
}

bool LowestCostRlFeatureTask::ComputeFeatures(
                      const NativeLowestCostContext& lowest_cost_context,
                      LowestCostRlInputData& lowest_cost_rl_input_data) {
  std::vector<float> input_0_list;      // 24 * 18
  std::vector<int64_t> input_1_list;    // 24 * 3
  std::vector<float> input_2_list;      // 24 * 1
  std::vector<float> input_3_list;      // 24 * 1
  std::vector<int64_t> input_4_list;    // 24 * 1
  std::vector<int32_t> input_5_list;    // 24 * 1
  int seq_length = 24;
  int state_dim = 18;
  int sparse_state_dim = 3;
  int action_dim = 1;
  int size = lowest_cost_context.rl_features().raw_features_size();
  if (size < 24) {
    return false;
  }
  // 滑动窗 求均值，方差
  std::vector<float> sliding_4_avg_bid_ratio_list(size, 0);
  std::vector<float> sliding_4_std_bid_ratio_list(size, 0);
  std::vector<float> sliding_10_avg_bid_ratio_list(size, 0);
  std::vector<float> sliding_10_std_bid_ratio_list(size, 0);
  SlidingWindowStatistics stats_4(4);     // 设置窗口大小为 4
  SlidingWindowStatistics stats_10(10);   // 设置窗口大小为 4
  for (int i = 0; i < size; i++) {
    auto bid_ratio = lowest_cost_context.rl_features().raw_features(i).bid_ratio();
    stats_4.addValue(bid_ratio);
    stats_10.addValue(bid_ratio);
    sliding_4_avg_bid_ratio_list[i] = stats_4.calculateMean();
    sliding_4_std_bid_ratio_list[i] = stats_4.calculateStdDev();
    sliding_10_avg_bid_ratio_list[i] = stats_10.calculateMean();
    sliding_10_std_bid_ratio_list[i] = stats_10.calculateStdDev();
  }
  // 特征归一化参数 Kconf
  std::vector<float> state_mean = {  9.62539237e-01,
        2.20414099e-01,
        5.04120844e-01,
        5.68520849e-01,
        2.29950224e-03,
        1.82294868e-03,
        7.53012728e-03,
        2.30645955e-03,
        1.82224574e-03,
        8.27614407e-03,
        2.31526255e-03,
        1.82889672e-03,
        4.04224336e+00,
        4.02430859e+00,
        9.29401276e-02,
        2.02129166e-01,
        9.01736723e-01,
        8.91582668e-01};    // NOLINT
  std::vector<float> state_std = {3.73493253e+00,
        1.85665593e+00,
        2.97309946e-01,
        3.18240862e-01,
        2.20468337e-03,
        4.17328986e-03,
        4.78569949e-01,
        1.43114898e-03,
        2.78121153e-03,
        2.92780611e-01,
        1.34477970e-03,
        2.38941102e-03,
        8.46083140e+00,
        8.39436539e+00,
        7.33701939e-01,
        1.12401601e+00,
        1.75298238e+00,
        1.41467936e+00};    // NOLINT
  int start = std::max(size - seq_length, 0);
  for (int i = start; i < size; i++) {
    auto past_pv_ratio = lowest_cost_context.rl_features().raw_features(i).past_pv_ratio();
    auto past_cost_ratio = lowest_cost_context.rl_features().raw_features(i).past_cost_ratio();
    auto bid_ratio = lowest_cost_context.rl_features().raw_features(i).bid_ratio();

    float pre_past_pv_ratio = lowest_cost_context.rl_features().raw_features(i).past_pv_ratio();    // NOLINT
    float pre_past_cost_ratio = lowest_cost_context.rl_features().raw_features(i).past_cost_ratio();    // NOLINT
    float pre_bid_ratio = lowest_cost_context.rl_features().raw_features(i).bid_ratio();

    float pre_4_past_pv_ratio = i-3 >= 0? lowest_cost_context.rl_features().raw_features(i - 3).past_pv_ratio(): 0;    // NOLINT
    float pre_4_past_cost_ratio = i-3 >= 0? lowest_cost_context.rl_features().raw_features(i - 3).past_cost_ratio(): 0;    // NOLINT
    float pre_4_bid_ratio = i-3 >= 0? lowest_cost_context.rl_features().raw_features(i - 3).bid_ratio(): 0;

    float pre_10_past_pv_ratio = i-9 >= 0? lowest_cost_context.rl_features().raw_features(i - 9).past_pv_ratio(): 0;    // NOLINT
    float pre_10_past_cost_ratio = i-9 >= 0? lowest_cost_context.rl_features().raw_features(i - 9).past_cost_ratio(): 0;    // NOLINT
    float pre_10_bid_ratio = i-9 >= 0? lowest_cost_context.rl_features().raw_features(i - 9).bid_ratio(): 0;

    float pre_23_past_pv_ratio = i-22 >= 0? lowest_cost_context.rl_features().raw_features(i - 22).past_pv_ratio(): 0;    // NOLINT
    float pre_23_past_cost_ratio = i-22 >= 0? lowest_cost_context.rl_features().raw_features(i - 22).past_cost_ratio(): 0;    // NOLINT

    int64_t pre_rn = lowest_cost_context.rl_features().raw_features(i).time_step();
    int64_t pre_4_rn = i-3 >= 0? lowest_cost_context.rl_features().raw_features(i - 3).time_step(): 0;
    int64_t pre_10_rn = i-9 >= 0? lowest_cost_context.rl_features().raw_features(i - 9).time_step(): 0;
    int64_t pre_23_rn = i-22 >= 0? lowest_cost_context.rl_features().raw_features(i - 22).time_step(): 0;

    // state
    auto p_error = lowest_cost_context.rl_features().raw_features(i).p_error();
    auto d_error = lowest_cost_context.rl_features().raw_features(i).d_error();
    float remaining_pv_ratio = 1.0 - pre_past_pv_ratio;
    float remaining_budget_ratio = 1.0 - pre_past_cost_ratio;
    std::cout << "goog rl feat: "
              << "index: " << pre_rn
              << "pre_past_pv_ratio: " << pre_past_pv_ratio
              << "remaining_pv_ratio: " << remaining_pv_ratio
              << "pre_past_cost_ratio: " << pre_past_cost_ratio
              << "remaining_budget_ratio" << remaining_budget_ratio
              << std::endl;
    float pv_ratio_three_speed = (pre_4_past_pv_ratio - pre_past_pv_ratio) * 1.0 / (pre_4_rn - pre_rn);
    float cost_ratio_three_speed = (pre_4_past_cost_ratio - pre_past_cost_ratio) * 1.0 / (pre_4_rn - pre_rn);
    float bid_ratio_three_speed = (pre_4_bid_ratio - pre_bid_ratio) * 1.0 / (pre_4_rn - pre_rn);
    float pv_ratio_nine_speed = (pre_10_past_pv_ratio - pre_past_pv_ratio) * 1.0 / (pre_10_rn - pre_rn);
    float cost_ratio_nine_speed = (pre_10_past_cost_ratio - pre_past_cost_ratio) * 1.0 / (pre_10_rn - pre_rn);
    float bid_ratio_nine_speed = (pre_10_bid_ratio - pre_bid_ratio) * 1.0 / (pre_10_rn - pre_rn);
    float pv_ratio_speed = (pre_23_past_pv_ratio - pre_past_pv_ratio) * 1.0 / (pre_23_rn - pre_rn);
    float cost_ratio_speed = (pre_23_past_cost_ratio - pre_past_cost_ratio) * 1.0 / (pre_23_rn - pre_rn);
    float sliding_4_avg_bid_ratio = sliding_4_avg_bid_ratio_list[i];
    float sliding_10_avg_bid_ratio = sliding_10_avg_bid_ratio_list[i];
    float sliding_4_std_bid_ratio = sliding_4_std_bid_ratio_list[i];
    float sliding_10_std_bid_ratio = sliding_10_std_bid_ratio_list[i];
    float alloc_ratio_three = pre_past_pv_ratio - pre_4_past_pv_ratio == 0? 1.0:
                      (pre_past_cost_ratio - pre_4_past_cost_ratio) / (pre_past_pv_ratio - pre_4_past_pv_ratio);    // NOLINT
    float alloc_ratio_ten = (pre_past_pv_ratio - pre_10_past_pv_ratio) == 0? 1.0:
                      (pre_past_cost_ratio - pre_10_past_cost_ratio) / (pre_past_pv_ratio - pre_10_past_pv_ratio);    // NOLINT
    std::cout << "goog rl feat: "
          << "index: " << pre_rn
          << "pre_past_pv_ratio: " << pre_past_pv_ratio
          << "remaining_pv_ratio: " << remaining_pv_ratio
          << "pre_past_cost_ratio: " << pre_past_cost_ratio
          << "pre_4_rn: " << pre_4_rn
          << "pre_10_rn: " << pre_10_rn
          << "remaining_budget_ratio: " << remaining_budget_ratio
          << "pv_ratio_three_speed: " << pv_ratio_three_speed
          << "cost_ratio_three_speed: " << cost_ratio_three_speed
          << "bid_ratio_three_speed: " << bid_ratio_three_speed
          << "pv_ratio_nine_speed: " << pv_ratio_nine_speed
          << "cost_ratio_nine_speed: " << cost_ratio_nine_speed
          << "bid_ratio_nine_speed: " << bid_ratio_nine_speed
          << "pv_ratio_speed: " << pv_ratio_speed
          << "cost_ratio_speed: " << cost_ratio_speed
          << "sliding_4_avg_bid_ratio: " << sliding_4_avg_bid_ratio
          << "sliding_10_avg_bid_ratio: " << sliding_10_avg_bid_ratio
          << "sliding_4_std_bid_ratio: " << sliding_4_std_bid_ratio
          << "sliding_10_std_bid_ratio: " << sliding_10_std_bid_ratio
          << "alloc_ratio_three: " << alloc_ratio_three
          << "alloc_ratio_ten: " << alloc_ratio_ten
          << "pre_4_past_pv_ratio: " << pre_4_past_pv_ratio
          << "pre_4_past_cost_ratio: " << pre_4_past_cost_ratio
          << "pre_4_bid_ratio: " << pre_4_bid_ratio
          << "pre_10_past_pv_ratio: " << pre_10_past_pv_ratio
          << "pre_10_past_cost_ratio: " << pre_10_past_cost_ratio
          << "pre_10_bid_ratio: "<< pre_10_bid_ratio
          << "pre_23_past_pv_ratio: " << pre_23_past_pv_ratio
          << "pre_23_past_cost_ratio: " << pre_23_past_cost_ratio
          << std::endl;

    p_error = (p_error - state_mean[0]) / state_std[0];
    d_error = (d_error - state_mean[1]) / state_std[1];
    remaining_pv_ratio = (remaining_pv_ratio - state_mean[2]) / state_std[2];
    remaining_budget_ratio = (remaining_budget_ratio - state_mean[3]) / state_std[3];
    pv_ratio_three_speed = (pv_ratio_three_speed - state_mean[4]) / state_std[4];
    cost_ratio_three_speed = (cost_ratio_three_speed - state_mean[5]) / state_std[5];
    bid_ratio_three_speed = (bid_ratio_three_speed - state_mean[6]) / state_std[6];
    pv_ratio_nine_speed = (pv_ratio_nine_speed - state_mean[7]) / state_std[7];
    cost_ratio_nine_speed = (cost_ratio_nine_speed - state_mean[8]) / state_std[8];
    bid_ratio_nine_speed = (bid_ratio_nine_speed - state_mean[9]) / state_std[9];
    pv_ratio_speed = (pv_ratio_speed - state_mean[10]) / state_std[10];
    cost_ratio_speed = (cost_ratio_speed - state_mean[11]) / state_std[11];
    sliding_4_avg_bid_ratio = (sliding_4_avg_bid_ratio - state_mean[12]) / state_std[12];
    sliding_10_avg_bid_ratio = (sliding_10_avg_bid_ratio - state_mean[13]) / state_std[13];
    sliding_4_std_bid_ratio = (sliding_4_std_bid_ratio - state_mean[14]) / state_std[14];
    sliding_10_std_bid_ratio = (sliding_10_std_bid_ratio - state_mean[15]) / state_std[15];
    float alloc_ratio_three_normal = (alloc_ratio_three - state_mean[16]) / state_std[16];
    float alloc_ratio_ten_normal = (alloc_ratio_ten - state_mean[17]) / state_std[17];
    
    std::vector<float> states = {p_error, d_error, remaining_pv_ratio, remaining_budget_ratio, pv_ratio_three_speed,    // NOLINT
                      cost_ratio_three_speed, bid_ratio_three_speed, pv_ratio_nine_speed, cost_ratio_nine_speed,    // NOLINT
                      bid_ratio_nine_speed, pv_ratio_speed, cost_ratio_speed, sliding_4_avg_bid_ratio, sliding_10_avg_bid_ratio,    // NOLINT
                      sliding_4_std_bid_ratio, sliding_10_std_bid_ratio, alloc_ratio_three_normal, alloc_ratio_ten_normal};
    input_0_list.insert(input_0_list.end(), std::make_move_iterator(states.begin()), std::make_move_iterator(states.end()));    // NOLINT
    // sparse_state
    int64_t author_industry = lowest_cost_context.author_industry();
    int64_t alloc_ratio_three_bucketed = 2;
    int64_t alloc_ratio_ten_bucketed = 2;
    float alloc_ratio_three_lower_thr = 0.25;
    float alloc_ratio_three_upper_thr = 5;
    float alloc_ratio_ten_lower_thr = 0.25;
    float alloc_ratio_ten_upper_thr = 5;
    if (alloc_ratio_three <= alloc_ratio_three_lower_thr) {
      alloc_ratio_three_bucketed = 0;
    } else if (alloc_ratio_three >= alloc_ratio_three_upper_thr) {
      alloc_ratio_three_bucketed = 1;
    } else {
    alloc_ratio_three_bucketed = 2;
    }
    if (alloc_ratio_ten <= alloc_ratio_ten_lower_thr) {
      alloc_ratio_ten_bucketed = 0;
    } else if (alloc_ratio_ten >= alloc_ratio_ten_upper_thr) {
      alloc_ratio_ten_bucketed = 1;
    } else {
      alloc_ratio_ten_bucketed = 2;
    }
    std::vector<int64_t> sparse_state = {author_industry, alloc_ratio_three_bucketed, alloc_ratio_ten_bucketed};    // NOLINT
    input_1_list.insert(input_1_list.end(), std::make_move_iterator(sparse_state.begin()), std::make_move_iterator(sparse_state.end()));    // NOLINT
    // action
    std::vector<float> action = {bid_ratio};
    input_2_list.insert(input_2_list.end(), std::make_move_iterator(action.begin()), std::make_move_iterator(action.end()));    // NOLINT
    // returns_to_go
    float return_reward_max = 580;
    double total_pred_cv = 0; //lowest_cost_context.total_pred_cv();
    if (false) {
      total_pred_cv = lowest_cost_context.rl_features().raw_features(i).total_pred_cv();
      if (0 > 0) {
        total_pred_cv = 0;
      }
    }
    std::vector<float> returns_to_go = {
      1.0f - static_cast<float>(total_pred_cv) / return_reward_max};
    input_3_list.insert(input_3_list.end(), std::make_move_iterator(returns_to_go.begin()), std::make_move_iterator(returns_to_go.end()));    // NOLINT
    // time_index
    int64_t index = lowest_cost_context.rl_features().raw_features(i).time_step();
    std::vector<int64_t> time_index = {index};
    input_4_list.insert(input_4_list.end(), std::make_move_iterator(time_index.begin()), std::make_move_iterator(time_index.end()));    // NOLINT
    // padding_mask
    std::vector<int32_t> padding_mask = {0};
    input_5_list.insert(input_5_list.end(), std::make_move_iterator(padding_mask.begin()), std::make_move_iterator(padding_mask.end()));    // NOLINT
  }

  lowest_cost_rl_input_data.input_0_list = std::move(input_0_list);
  lowest_cost_rl_input_data.input_1_list = std::move(input_1_list);
  lowest_cost_rl_input_data.input_2_list = std::move(input_2_list);
  lowest_cost_rl_input_data.input_3_list = std::move(input_3_list);
  lowest_cost_rl_input_data.input_4_list = std::move(input_4_list);
  lowest_cost_rl_input_data.input_5_list = std::move(input_5_list);
  return true;
}

template<typename T>
void prinf_data(std::vector<T>& data, std::string name) {
    std::string out = name + "\n = [";
    for (auto& d : data) {
        out += std::to_string(d) + ", ";
    }
    out += "]";
    std::cout << out << std::endl;
}


int main() {
    NativeLowestCostContext lowest_cost_context;
    LowestCostRlInputData lowest_cost_rl_input_data;
    LowestCostRlFeatureTask lowest_cost_rl_feature_task;
    lowest_cost_rl_feature_task.PushFeatures(&lowest_cost_context);
    lowest_cost_rl_feature_task.ComputeFeatures(lowest_cost_context, lowest_cost_rl_input_data);
    prinf_data(lowest_cost_rl_input_data.input_0_list, "input_0_list");
    prinf_data(lowest_cost_rl_input_data.input_1_list, "input_1_list");
    prinf_data(lowest_cost_rl_input_data.input_2_list, "input_2_list");
    prinf_data(lowest_cost_rl_input_data.input_3_list, "input_3_list");
    prinf_data(lowest_cost_rl_input_data.input_4_list, "input_4_list");
    prinf_data(lowest_cost_rl_input_data.input_5_list, "input_5_list");
    std::cout << "hello world" << std::endl;
}
