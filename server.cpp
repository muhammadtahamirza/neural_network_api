#include "httplib.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "neural_network.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//  convert any image to MNIST format (28x28 grayscale)
Eigen::VectorXd convertToMNISTFormat(const std::string& image_data) {
    int width, height, channels;

    // Load as grayscale but keep original channels to detect background
    unsigned char* image = stbi_load_from_memory(
        reinterpret_cast<const unsigned char*>(image_data.data()),
        image_data.size(),
        &width, &height, &channels,
        0 // Keep original channels to detect background
    );

    if (!image) throw std::runtime_error("Failed to load image");

    double total_brightness = 0.0;
    for (int i = 0; i < width * height * channels; i += channels) {
        total_brightness += image[i]; // Use first channel (grayscale or R)
    }
    double avg_brightness = total_brightness / (width * height);

    bool should_invert = avg_brightness > 127; // If background is bright, invert

    Eigen::VectorXd mnist_image(28 * 28);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int src_x = static_cast<int>(x * static_cast<double>(width) / 28.0);
            int src_y = static_cast<int>(y * static_cast<double>(height) / 28.0);
            src_x = std::min(src_x, width - 1);
            src_y = std::min(src_y, height - 1);

            // Get pixel value (handle both grayscale and color)
            unsigned char pixel;
            if (channels == 1) {
                pixel = image[src_y * width + src_x];
            }
            else {
                // Convert color to grayscale: 0.299R + 0.587G + 0.114B
                int idx = (src_y * width + src_x) * channels;
                pixel = static_cast<unsigned char>(
                    0.299 * image[idx] +
                    0.587 * image[idx + 1] +
                    0.114 * image[idx + 2]
                    );
            }

            double normalized = static_cast<double>(pixel) / 255.0;

            // Apply inversion only if background is bright
            if (should_invert) {
                normalized = 1.0 - normalized;
            }

            mnist_image(y * 28 + x) = normalized;
        }
    }

    stbi_image_free(image);
    return mnist_image;
}

int predictDigit(Eigen::VectorXd& mnist_image) {
    std::vector<layer> layers;
    layers.emplace_back(784, 64); //hidden layer
    layers.emplace_back(64, 10); //ouput layer



    neural_network network(layers);
   std:: cout << "layers connected to network successfully !!!\n\n";


    std::cout << "loading model ...\n";
    network.load_model("taha_model.bin");
    return network.predict(mnist_image);

} 
int main() {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    httplib::Server server;

    // Enable CORS
    auto enable_cors = [](httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        };

    // Handle preflight requests
    server.Options("/predict", [enable_cors](const httplib::Request& req, httplib::Response& res) {
        enable_cors(res);
        res.status = 200;
        });

    // Main prediction endpoint
    server.Get("/", [enable_cors](const httplib::Request& req, httplib::Response& res) {
        enable_cors(res);

        // Simple test response
        res.set_content(
            "{\"status\": \"server_is_working\", \"message\": \"MNIST Digit Recognition API is running!\", \"endpoints\": [\"GET /test\", \"POST /predict\"]}",
            "application/json"
        );

        std::cout << "✅ Test endpoint called\n";
        });

    server.Post("/predict", [enable_cors](const httplib::Request& req, httplib::Response& res) {
        enable_cors(res);

        try {
            // Check if request contains image data
            if (req.body.empty()) {
                res.status = 400;
                res.set_content("{\"error\": \"No image data provided\"}", "application/json");
                return;
            }

            std::cout << "📨 Received image of size: " << req.body.size() << " bytes\n";

            // Convert image to MNIST format
            Eigen::VectorXd mnist_image = convertToMNISTFormat(req.body);
 
            std::cout << "✅ Converted to MNIST format (28x28 grayscale)\n";

            // Get prediction from your model
            int prediction = predictDigit(mnist_image);

            std::cout << "🔮 Predicted digit: " << prediction << "\n";
            std::cout << "********************************************\n\n";
            // Return prediction as JSON
            std::cout.flush();
            res.set_content(
                "{\"prediction\": " + std::to_string(prediction) + "}",
                "application/json"
            );

        }
        catch (const std::exception& e) {
            std::cerr << "❌ Error: " << e.what() << "\n";
            res.status = 500;
            res.set_content(
                "{\"error\": \"" + std::string(e.what()) + "\"}",
                "application/json"
            );
        }
        });

    std::cout << "🚀 Server started on http://localhost:8080\n";
    std::cout << "📋 Endpoint: POST /predict (send image data in request body)\n";

    server.listen("localhost", 8080);
    return 0;

}
