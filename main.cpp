#include "windmill.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vector>

using namespace std;
using namespace cv;

// 代价函数的结构体
struct WindmillResidual {
  WindmillResidual(double dt, double observed_angle)
      : dt_(dt), observed_angle_(observed_angle) {}

  template <typename T>
  bool operator()(const T *const params, T *residual) const {
    T A0 = params[0];
    T A = params[1];
    T w = params[2];
    T phi = params[3];

    T dangle =
        A0 * T(dt_) + (A / w) * (ceres::cos(phi + T(1.57)) -
                                 ceres::cos(w * (T(dt_)) + phi + T(1.57)));

    residual[0] = cos(dangle) - T(observed_angle_);
    return true;
  }

private:
  const double dt_;
  const double observed_angle_;
};

// 定义真值
const double true_A = 0.785;
const double true_omega = 1.884;
const double true_phi = 0.24;
const double true_A0 = 1.305;

int main() {

  std::vector<double> timings;
  int run = 0;
   while (run < 10) {
    cout << "Outer Iteration: " << run + 1 << endl;  // 输出外层循环的当前迭代次数
    std::chrono::milliseconds t =std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    WINDMILL::WindMill wm(t.count());
    double tstart = (double)t.count();
    cv::Mat src;

    std::vector<double> time_data;
    std::vector<double> angle_data;

    // Load template image for "R" mark
    Mat templateR = imread("../image/R.png", IMREAD_GRAYSCALE);
    if (templateR.empty()) {
      cerr << "Error: Could not load template image for 'R' mark!" << endl;
      return -1;
    }

    // Load template image for "Hammer" mark
    Mat templateHammer = imread("../image/target.png", IMREAD_GRAYSCALE);
    if (templateHammer.empty()) {
      cerr << "Error: Could not load template image for 'Hammer' mark!" << endl;
      return -1;
    }

    // 构建问题
    ceres::Problem problem;
    // 设置初始值并进行拟合
    double initial_A = true_A + 1; // 更接近真实值的初始参数
    double initial_omega = true_omega + 1;
    double initial_phi = true_phi + 1;
    double initial_A0 = true_A0 + 1;

    double params[4] = {initial_A0, initial_A, initial_omega, initial_phi};

    // 将参数块添加到问题中
    problem.AddParameterBlock(params, 4);
    int iteration_count = 0;  // 初始化迭代计数器
    while (1) {
      iteration_count++;  // 增加迭代计数器
      cout << "Inner Iteration: " << iteration_count << endl;  // 输出当前迭代次数
      cout << "Outer Iteration: " << run + 1 << endl;  // 输出外层循环的当前迭代次数
      t = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch());
      double current_time = static_cast<double>(t.count());
      src = wm.getMat((double)t.count());
      auto start = std::chrono::high_resolution_clock::now();
      // Convert frame to grayscale
      Mat graySrc;
      cvtColor(src, graySrc, COLOR_BGR2GRAY);

      // Template matching to find the "R" mark
      Mat result;
      matchTemplate(graySrc, templateR, result, TM_CCOEFF_NORMED);
      double minVal_R, maxVal_R;
      Point minLoc_R, maxLoc_R;
      minMaxLoc(result, &minVal_R, &maxVal_R, &minLoc_R, &maxLoc_R);

      Point2d centerR(maxLoc_R.x + templateR.cols / 2,
                      maxLoc_R.y + templateR.rows / 2);
      // 二值化
      Mat binaryTemplate, binaryTarget;
      threshold(templateHammer, binaryTemplate, 64, 255, THRESH_BINARY);
      threshold(graySrc, binaryTarget, 64, 255, THRESH_BINARY);

      // 查找轮廓
      vector<vector<Point>> contoursTemplate, contoursTarget;
      findContours(binaryTemplate, contoursTemplate, RETR_EXTERNAL,
                   CHAIN_APPROX_SIMPLE);
      findContours(binaryTarget, contoursTarget, RETR_EXTERNAL,
                   CHAIN_APPROX_SIMPLE);
      // 计算模板轮廓的面积
      double templateArea = contourArea(contoursTemplate[0]);
      Point2d vectorPR;
      // 遍历目标图像中的所有轮廓
      for (size_t i = 0; i < contoursTarget.size(); i++) {
        double targetArea = contourArea(contoursTarget[i]);
        // 如果轮廓面积与模板轮廓面积相近，则认为匹配
        if (fabs(targetArea - templateArea) < 0.1 * templateArea) {
          // 绘制匹配的轮廓
          drawContours(src, contoursTarget, (int)i, Scalar(0, 255, 0), 2);

          // 计算匹配轮廓的质心
          Moments m = moments(contoursTarget[i]);
          Point2d center = Point2f(m.m10 / m.m00, m.m01 / m.m00);
          vectorPR = center - centerR;
          // 绘制质心
          // circle(src, center, 5, Scalar(255, 0, 0), -1);
/* 
          // 输出质心坐标
          cout << "Matched object center: (" << center.x << ", " << center.y
               << ")" << endl; */
          break;
        }
      }

      Point2d normvectorPR = vectorPR / norm(vectorPR);
      double x = normvectorPR.x;
      // cout<<x<<endl;
      //  记录时间和角度数据
      double tend = (double)t.count();
      double dt = (tend - tstart) /1000;
      // 计算匹配区域的中心点
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<WindmillResidual, 1, 4>(
              new WindmillResidual(dt, x)),
          nullptr, params);
      ceres::Solver::Options options;
      options.max_num_iterations = 1000;
      options.linear_solver_type = ceres::DENSE_QR;
      // options.minimizer_progress_to_stdout = true; // 禁用进度输出
      // 设置参数边界
      problem.SetParameterLowerBound(params, 0, 0.5);  // A0 >= 0
      problem.SetParameterUpperBound(params, 0, 5.0);  // A0 <= 10
      problem.SetParameterLowerBound(params, 1, 0.5);  // A >= 0
      problem.SetParameterUpperBound(params, 1, 5.0);  // A <= 10
      problem.SetParameterLowerBound(params, 2, 1.0);  // omega >= 0
      problem.SetParameterUpperBound(params, 2, 5.0);  // omega <= 10
      problem.SetParameterLowerBound(params, 3, 0.2);  // phi >= -pi
      problem.SetParameterUpperBound(params, 3, 3.14); // phi <= pi

      // 运行求解器
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      // 输出拟合结果和误差
      // cout << "Run " << run + 1 << " Results:" << endl;
      cout << "A0: " << params[0] << ", true_A0: " << true_A0 << endl;
      cout << "A: " << params[1] << ", true_A: " << true_A << endl;
      cout << "omega: " << params[2] << ", true_omega: " << true_omega << endl;
      cout << "phi: " << params[3] << ", true_phi: " << true_phi
           << endl; // cout << summary.BriefReport() << endl;
      // 检查拟合结果是否收敛
      bool converged = std::abs(params[0] - true_A0) < 0.05 * true_A0 &&
                       std::abs(params[1] - true_A) < 0.05 * true_A &&
                       std::abs(params[2] - true_omega) < 0.05 * true_omega &&
                       std::abs(params[3] - true_phi) < 0.05 * true_phi;

      if (converged) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        timings.push_back(elapsed.count());
        run++;
        break;
      } else {
      }
    }
  }

  // 计算并输出平均时间
  double average_time =
      std::accumulate(timings.begin(), timings.end(), 0.0) / 10;
  cout << "Average time: " << average_time << "seconds" << endl;

  return 0;
}
