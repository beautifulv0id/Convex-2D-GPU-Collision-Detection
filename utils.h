#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;

int get_num_batches_in_dir(std::string directoryPath){
    // Specify the file extension pattern
    std::string fileExtension = ".npy";

    // Initialize a counter for the matching files
    int fileCount = 0;

    // Iterate over the files in the directory
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (fs::is_regular_file(entry) && entry.path().extension() == fileExtension) {
            try{
                int batchNum = std::stoi(entry.path().filename().string());
                fileCount++;
            } catch(...){
                continue;
            }
        }
    }

    return fileCount;
}