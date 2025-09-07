#include "dataset.h"


uint32_t swap_endian(uint32_t value) {
    return ((value & 0x000000FF) << 24) |
        ((value & 0x0000FF00) << 8) |
        ((value & 0x00FF0000) >> 8) |
        ((value & 0xFF000000) >> 24);
}
using namespace std;

namespace dataset {
    void read_mnist_train_data(const std::string& path,
        std::vector< Eigen::VectorXd>& data) {
        std::ifstream file(path, std::ios::binary);
        cout << "opening file for images ...\n";
        if (file.is_open()) {
            cout << "loading images ...\n";
            int magic_number = 0;
            int number_of_images = 0;
            int rows = 0;
            int cols = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = swap_endian(magic_number);
            file.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = swap_endian(number_of_images);
            file.read((char*)&rows, sizeof(rows));
            rows = swap_endian(rows);
            file.read((char*)&cols, sizeof(cols));
            cols = swap_endian(cols);




            for (int i = 0; i < number_of_images; i++) {
                Eigen::VectorXd vec(rows * cols);
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        unsigned char temp = 0;
                        file.read((char*)&temp, sizeof(temp));
                        vec(r * cols + c) = (double)temp / 255.0;

                    }
                }
                data.push_back(vec);
            }
             std::cout << "Images loaded Successfully  !!!\n\n";
        }
        else {
            throw "file not exist for images";
        }
    }

    void read_mnist_train_label(const std::string& path,
        std::vector<Eigen::VectorXd>& data) {

        std::ifstream file(path, std::ios::binary);
        cout << "opening file for labels ...\n";
        if (file.is_open()) {
            cout << "loading labels ...\n";
            int magic_number = 0;
            int number_of_items = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = swap_endian(magic_number);
            file.read((char*)&number_of_items, sizeof(number_of_items));
            number_of_items = swap_endian(number_of_items);

            for (int i = 0; i < number_of_items; i++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                Eigen::VectorXd vec(10);
                vec.setZero();
                vec((int)temp) = 1.0;
                data.push_back(vec);
            }
            cout << "labels loaded successfully !!! \n\n";
        }
        else {
            throw "file not exist for labels";
        }

    }

    void read_mnist_test_data(const std::string& path,
        std::vector<Eigen::VectorXd>& data) {
        std::ifstream file(path, std::ios::binary);
        cout << "opening test data images ...\n";
        if (file.is_open()) {
            cout << "loading test data ...\n";
            int magic_number = 0;
            int number_of_images = 0;
            int rows = 0;
            int cols = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = swap_endian(magic_number);
            file.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = swap_endian(number_of_images);
            file.read((char*)&rows, sizeof(rows));
            rows = swap_endian(rows);
            file.read((char*)&cols, sizeof(cols));
            cols = swap_endian(cols);

            for (int i = 0; i < number_of_images; i++) {
                Eigen::VectorXd vec(rows * cols);
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        unsigned char temp = 0;
                        file.read((char*)&temp, sizeof(temp));
                        vec(r * cols + c) = (double)temp / 255.0;
                    }
                }
                data.push_back(vec);
            }
            cout << "loaded successfullu !!! \n";
        }
    }

    void read_mnist_test_label(const std::string& path,
        std::vector<Eigen::VectorXd>& data) {
        std::ifstream file(path, std::ios::binary);
        cout << "opneing test laels ...\n";
        if (file.is_open()) {
            cout << "loading labels ...\n";
            int magic_number = 0;
            int number_of_items = 0;

            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = swap_endian(magic_number);
            file.read((char*)&number_of_items, sizeof(number_of_items));
            number_of_items = swap_endian(number_of_items);

            for (int i = 0; i < number_of_items; i++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                Eigen::VectorXd vec(10);
                vec.setZero();
                vec((int)temp) = 1.0;
                data.push_back(vec);
            }
            cout << "labels loaded successfully !!!\n";
        }
    }
}
