#include <string>
#include <vector>
#include <algorithm>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

struct Arguments {
    std::string data_dir = "./data/";
    int num_batches = 100;
    int batch_size = 100000;
    int start_batch_count = 0;
    int variance_dim = 5;
    int pose_dim = 3;
    int num_poses = 64*64*64*64;
    int num_variances = 64*64*64*64;
    int max_samples = 4000000;
    std::vector<float> min_variance = {0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> max_variance = {0.3, 0.3, 0.3, 0.3, 0.3};
    std::vector<float> min_pose = {0.1, 0.1, 0.0};
    std::vector<float> max_pose = {5, 5, 2*M_PI};
    std::vector<float> accuracy_bins = {0.0, 0.001, 0.1, 1.0};
    std::vector<float> bin_accuracy = {0.0001, 0.001, 0.01, 0};
    float robot_width = 4.07;
    float robot_height = 1.74;
};

Arguments parse_args(int argc, char** argv) {
    Arguments a;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("data_dir", po::value<std::string>(), "where to store the data")
        ("num_batches,n", po::value<int>(), "number of batches")
        ("batch_size,b", po::value<int>(), "number of samples per batch")
        ("start_batch_count,s", po::value<int>(), "start value for batches")
        ("variance_dim", po::value<int>(), "dimension of variance")
        ("pose_dim", po::value<int>(), "dimension of pose")
        ("num_poses", po::value<int>(), "number of poses")
        ("num_variances", po::value<int>(), "number of variances")
        ("max_samples", po::value<int>(), "maximum number of samples for z-test")
        ("accuracy_bins", po::value<std::vector<float>>()->multitoken(), "accuracy bins e.g. [0.0001 0.001 0.01 0]")
        ("bin_accuracy", po::value<std::vector<float>>()->multitoken(), "accuracy for each bin e.g. [0.0001, 0.001, 0.01, 0]")
        ("min_variance", po::value<std::vector<float>>()->multitoken(), "min variance for each dimension e.g. [0.0, 0.0, 0.0, 0.0, 0.0]")
        ("max_variance", po::value<std::vector<float>>()->multitoken(), "max variance for each dimension e.g. [0.3, 0.3, 0.3, 0.3, 0.3]")
        ("min_pose", po::value<std::vector<float>>()->multitoken(), "min pose for each dimension e.g. [0.1, 0.1, 0.0]")
        ("max_pose", po::value<std::vector<float>>()->multitoken(), "max pose for each dimension e.g. [5, 5, 2*M_PI]")
        ("robot_width,w", po::value<float>(), "robot width")
        ("robot_height,h", po::value<float>(), "robot height");
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(1);
    }

    if(vm.count("data_dir")) {
        a.data_dir = vm["data_dir"].as<std::string>();
    }
    if(vm.count("num_batches")) {
        a.num_batches = vm["num_batches"].as<int>();
    }
    if(vm.count("batch_size")) {
        a.batch_size = vm["batch_size"].as<int>();
    }
    if(vm.count("start_batch_count")) {
        a.start_batch_count = vm["start_batch_count"].as<int>();
    }
    if(vm.count("variance_dim")) {
        a.variance_dim = vm["variance_dim"].as<int>();
    }
    if(vm.count("pose_dim")) {
        a.pose_dim = vm["pose_dim"].as<int>();
    }
    if(vm.count("num_poses")) {
        a.num_poses = vm["num_poses"].as<int>();
    }
    if(vm.count("num_variances")) {
        a.num_variances = vm["num_variances"].as<int>();
    }
    if(vm.count("max_samples")) {
        a.max_samples = vm["max_samples"].as<int>();
    }
    if (vm.count("accuracy_bins")) {
        std::vector<float> values = vm["accuracy_bins"].as<std::vector<float>>();
        a.accuracy_bins = values;
    }
    if (vm.count("bin_accuracy")) {
        std::vector<float> values = vm["bin_accuracy"].as<std::vector<float>>();
        a.bin_accuracy = values;
    }
    if (vm.count("min_variance")) {
        std::vector<float> values = vm["min_variance"].as<std::vector<float>>();
        a.min_variance = values;
    }
    if (vm.count("max_variance")) {
        std::vector<float> values = vm["max_variance"].as<std::vector<float>>();
        a.max_variance = values;
    }
    if (vm.count("min_pose")) {
        std::vector<float> values = vm["min_pose"].as<std::vector<float>>();
        a.min_pose = values;
    }
    if (vm.count("max_pose")) {
        std::vector<float> values = vm["max_pose"].as<std::vector<float>>();

        a.max_pose = values;
    }
    if(vm.count("robot_width")) {
        a.robot_width = vm["robot_width"].as<float>();
    }
    if(vm.count("robot_height")) {
        a.robot_height = vm["robot_height"].as<float>();
    }
    return a;
}