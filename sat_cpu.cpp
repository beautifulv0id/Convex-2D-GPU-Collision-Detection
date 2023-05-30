#include <chrono>
#include <cmath>
#include <iostream>
#include <npy.hpp>
#include <random>
#include <vector>

std::vector<std::vector<double>>
generateQueryPoints(double height_r, double width_r,
                    const std::vector<std::vector<double>> &borders,
                    int num_queries) {
  std::vector<std::vector<double>> points;
  points.reserve(num_queries);

  std::random_device rd;
  std::mt19937 generator(rd());

  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  double r_offset = (width_r + height_r) / 4.0;

  for (int i = 0; i < num_queries; ++i) {
    std::vector<double> pose_o;
    pose_o.reserve(borders.size());

    for (const auto &border : borders) {
      double pose =
          uniform_dist(generator) * (border[1] - border[0]) + border[0];
      pose_o.push_back(pose);
    }

    double theta_r = uniform_dist(generator) * 2.0 * M_PI;

    double x_r = std::cos(theta_r) * (pose_o[0] + r_offset) +
                 normal_dist(generator) *
                     (pose_o[0] / (2.0 * 1.96) + pose_o[2] + pose_o[5]);
    double y_r = std::sin(theta_r) * (pose_o[1] + r_offset) +
                 normal_dist(generator) *
                     (pose_o[1] / (2.0 * 1.96) + pose_o[3] + pose_o[6]);

    std::vector<double> point = {x_r, y_r, theta_r, width_r, height_r};
    point.insert(point.end(), pose_o.begin(), pose_o.end());

    points.push_back(point);
  }

  return points;
}

std::vector<std::vector<double>>
getCornerPoints(double x, double y, double theta, double width, double height) {
  std::vector<std::vector<double>> corners(2, std::vector<double>(4));
  corners[0][0] = x - width / 2;
  corners[0][1] = x + width / 2;
  corners[0][2] = x + width / 2;
  corners[0][3] = x - width / 2;
  corners[1][0] = y - height / 2;
  corners[1][1] = y - height / 2;
  corners[1][2] = y + height / 2;
  corners[1][3] = y + height / 2;

  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);

  for (int i = 0; i < 4; ++i) {
    double x_rot = corners[0][i] * cos_theta - corners[1][i] * sin_theta;
    double y_rot = corners[0][i] * sin_theta + corners[1][i] * cos_theta;
    corners[0][i] = x_rot;
    corners[1][i] = y_rot;
  }

  return corners;
};

bool isSeparatingAxis(const std::vector<double> &o,
                      const std::vector<std::vector<double>> &p1,
                      const std::vector<std::vector<double>> &p2) {
  double min1 = std::numeric_limits<double>::infinity();
  double max1 = -std::numeric_limits<double>::infinity();
  double min2 = std::numeric_limits<double>::infinity();
  double max2 = -std::numeric_limits<double>::infinity();

  for (const auto &v : p1) {
    double projection = v[0] * o[0] + v[1] * o[1];
    min1 = std::min(min1, projection);
    max1 = std::max(max1, projection);
  }

  for (const auto &v : p2) {
    double projection = v[0] * o[0] + v[1] * o[1];
    min2 = std::min(min2, projection);
    max2 = std::max(max2, projection);
  }

  return max1 < min2 || max2 < min1;
};

bool collideSAT(std::vector<std::vector<double>> &r,
                std::vector<std::vector<double>> &o) {
  for (int i = 0; i < 4; ++i) {
    std::vector<double> er = {r[0][(i + 1) % 4] - r[0][i],
                              r[1][(i + 1) % 4] - r[1][i]};
    if (isSeparatingAxis(er, r, o)) {
      return false;
    }
    std::vector<double> eo = {o[0][(i + 1) % 4] - o[0][i],
                              o[1][(i + 1) % 4] - o[1][i]};
    if (isSeparatingAxis(eo, r, o)) {
      return false;
    }
  }
  return true;
}

bool sample(std::vector<double> q) {
  const auto x_r = q[0];
  const auto y_r = q[1];
  const auto theta_r = q[2];
  const auto width_r = q[3];
  const auto height_r = q[4];
  const auto mu_width_o = q[5];
  const auto mu_height_o = q[6];
  const auto var_x_o = q[7];
  const auto var_y_o = q[8];
  const auto var_theta_o = q[9];
  const auto var_height_o = q[10];
  const auto var_width_o = q[11];

  auto p_r = getCornerPoints(x_r, y_r, theta_r, width_r, height_r);

  // generate obstacle
  std::random_device rd;
  std::mt19937 generator(rd());

  double min_height = 0.00001;
  double min_width = 0.00001;
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  // TODO check if it has to be std::sqrt(var_x_o)
  double x_o = normal_dist(generator) * var_x_o;
  double y_o = normal_dist(generator) * var_y_o;
  double theta_o = normal_dist(generator) * var_theta_o;
  double width_o =
      std::max(min_width, normal_dist(generator) * var_width_o + mu_width_o);
  double height_o =
      std::max(min_height, normal_dist(generator) * var_height_o + mu_height_o);

  auto p_o = getCornerPoints(x_o, y_o, theta_o, width_o, height_o);
  return collideSAT(p_r, p_o);
}

double calcSlack(double alpha, double nsamples, double nsamples_true) {
  if (alpha != 0.05) {
    throw std::invalid_argument("Significance level " + std::to_string(alpha) +
                                " is not supported");
  }

  double z = 1.645;

  if (nsamples_true == nsamples || nsamples_true == 0) {
    return std::log(1.0 / alpha) / nsamples;
  } else {
    return z / nsamples *
           std::sqrt(nsamples_true - nsamples_true * nsamples_true / nsamples);
  }
}

double pr_ci(std::vector<double> query, double alpha, int n_samples_init,
             int n_samples_per_batch, int n_samples_max,
             const std::vector<double> &accuracy_bins,
             const std::vector<double> &bin_slack) {

  double k_total = 0;
  double k_true = 0;
  int b = 0;

  while (true) {
#pragma omp parallel for reduction(+ : k_total, k_true)
    for (int i = 0; i < (b == 0 ? n_samples_init : n_samples_per_batch); ++i) {
      bool result = sample(query);
      if (result) {
        k_true += 1.0;
      }
      k_total += 1.0;
    }

    b += 1;

    double slack = calcSlack(alpha, k_total, k_true);
    double p = k_true / k_total;

    for (int i = 0; i < accuracy_bins.size() - 1; ++i) {
      if (p >= accuracy_bins[i] && p < accuracy_bins[i + 1] &&
          slack < bin_slack[i]) {
        return p;
      }
    }

    if (k_total >= n_samples_max) {
      return p;
    }
  }
}

std::vector<double> computeCollisionProbabilities(
    const std::vector<std::vector<double>> &queries, double alpha,
    int n_samples_init, int n_samples_per_batch, int n_samples_max,
    const std::vector<double> &accuracy_bins,
    const std::vector<double> &bin_slack,
    std::vector<std::chrono::duration<double>> &elapsed_times) {
  std::vector<double> collision_probabilities;
  collision_probabilities.reserve(queries.size());

  elapsed_times.reserve(queries.size());

  for (const auto &q : queries) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto p = pr_ci(q, alpha, n_samples_init, n_samples_per_batch, n_samples_max,
                   accuracy_bins, bin_slack);
    collision_probabilities.push_back(p);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    elapsed_times.push_back(elapsed_time);
  }

  return collision_probabilities;
}

void testSAT() {
  auto p1 = getCornerPoints(1.64591815, 0.35429279, 0.49918012, 0.75, 0.8);
  std::cout << p1[0][0] << ", " << p1[1][0] << std::endl;
  std::cout << p1[0][1] << ", " << p1[1][1] << std::endl;
  std::cout << p1[0][2] << ", " << p1[1][2] << std::endl;
  std::cout << p1[0][3] << ", " << p1[1][3] << std::endl;

  std::cout << std::endl;

  // auto p2 = getCornerPoints(1.02437856, 1.00520859, 0.28340416, 0.75, 0.8);
  auto p2 = getCornerPoints(1.02437856, 1.00520859, 0.28340416, 0.75, 0.8);
  std::cout << p2[0][0] << ", " << p2[1][0] << std::endl;
  std::cout << p2[0][1] << ", " << p2[1][1] << std::endl;
  std::cout << p2[0][2] << ", " << p2[1][2] << std::endl;
  std::cout << p2[0][3] << ", " << p2[1][3] << std::endl;

  auto collision = collideSAT(p1, p2);

  std::cout << "Collision: " << collision << std::endl;
}

int main(int argc, char *argv[]) {
  srand(static_cast<unsigned>(time(0)));

  int num_queries = 200;

  double height_r = 0.8;
  double width_r = 0.75;

  std::vector<std::vector<double>> borders = {
      {0.5, 3.0},   // mu_width_o
      {0.5, 3.0},   // mu_height_o
      {0.001, 0.3}, // var_x_o
      {0.001, 0.3}, // var_y_o
      {0.001, 0.3}, // var_theta_o
      {0.001, 0.3}, // var_width_o
      {0.001, 0.3}, // var_height_o
  };

  std::cout << "Generate " << num_queries << " query points" << std::endl;
  std::vector<std::vector<double>> queries =
      generateQueryPoints(height_r, width_r, borders, num_queries);

  // std::cout << "Query points:" << std::endl;
  // for (const auto &query : queries) {
  //   std::cout << query[0] << ", " << query[1] << ", " << query[2]
  //             << ", " << query[3] << ", " << query[4] << ", "
  //             << query[5] << ", " << query[6] << ", " << query[7]
  //             << ", " << query[8] << ", " << query[9] << ", "
  //             << query[10] << ", " << query[11] << ", " << std::endl;
  // }

  int n_samples_max = 4000000;
  double alpha = 0.05;
  int n_samples_init = 500;
  int n_samples_per_batch = 100;
  std::vector<double> accuracy_bins = {0.0, 0.001, 0.01, 0.01, 1.0};
  // std::vector<double> accuracy_bins = {0.0, 1.0};
  std::vector<double> bin_slack = {0.00005, 0.0005, 0.001, 0.01};
  // std::vector<double> bin_slack = {0.0};

  std::cout << "Compute collision probabilities" << std::endl;

  std::vector<std::chrono::duration<double>> elapsed_times;

  std::vector<double> collisionProbabilities = computeCollisionProbabilities(
      queries, alpha, n_samples_init, n_samples_per_batch, n_samples_max,
      accuracy_bins, bin_slack, elapsed_times);

  std::chrono::duration<double> sum =
      std::accumulate(elapsed_times.begin(), elapsed_times.end(),
                      std::chrono::duration<double>(0));

  std::cout << "Elapsed time: " << sum.count() << " seconds" << std::endl;

  std::cout << "Elapsed time per query: "
            << sum.count() / collisionProbabilities.size() << " seconds"
            << std::endl;

  std::cout << "Collision probabilities:" << std::endl;

  size_t elapsed_times_shape[1] = {(size_t)num_queries};
  std::vector<double> elapsed_times_seconds;
  for (const auto &elapsed_time : elapsed_times) {
    elapsed_times_seconds.push_back(elapsed_time.count());
  }

  std::string data_dir = "stats/ztest/speed/";
  npy::SaveArrayAsNumpy(data_dir + std::string("cpu.npy"), false, 1, elapsed_times_shape, elapsed_times_seconds);

  std::cout << "Saved elapsed times to " << data_dir << std::endl;

  double p_mean = 0;
  for (int i = 0; i < collisionProbabilities.size(); i++) {
    //   std::cout << "Query " << i << ": " << std::endl;
    //   std::cout << "x_o: " << queries[i][0] << std::endl;
    //   std::cout << "y_o: " << queries[i][1] << std::endl;
    //   std::cout << "t_o: " << queries[i][2] << std::endl;
    //   std::cout << "mu_width_o: " << queries[i][5] << std::endl;
    //   std::cout << "mu_height_o: " << queries[i][6] << std::endl;
    //   std::cout << "var_x_o: " << queries[i][7] << std::endl;
    //   std::cout << "var_y_o: " << queries[i][8] << std::endl;
    //   std::cout << "var_theta_o: " << queries[i][9] << std::endl;
    //   std::cout << "var_height_o: " << queries[i][10] << std::endl;
    //   std::cout << "var_width_o: " << queries[i][11] << std::endl;
    //   std::cout << "p: " << collisionProbabilities[i] << std::endl;
    p_mean += collisionProbabilities[i];
  }
  std::cout << "Mean collision probability: " << p_mean / num_queries
            << std::endl;

  return 0;
}
