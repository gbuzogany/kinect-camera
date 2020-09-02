// based on: https://gist.github.com/DrWateryCat/c225e436d5761bbb21fd622c78e78972
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>

#include <libusb-1.0/libusb.h>
#include <libfreenect/libfreenect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <pthread.h>
#include <unistd.h>
#include <csignal>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

static volatile bool die = false;

int angle = 0;

class myMutex {
public:
    myMutex() {
        pthread_mutex_init( &m_mutex, NULL );
    }
    void lock() {
        pthread_mutex_lock( &m_mutex );
    }
    void unlock() {
        pthread_mutex_unlock( &m_mutex );
    }
private:
    pthread_mutex_t m_mutex;
};

class Kinect : public Freenect::FreenectDevice{
private:
    std::vector<uint8_t> m_color_buffer;
    std::vector<uint16_t> m_gamma;
    
    cv::Mat m_rgbMat;
    cv::Mat m_ownMat;
    
    myMutex m_rgbMutex;
    
    bool m_new_rgb_frame = false;
    
public:
    Kinect(freenect_context* ctx, int index)
    :Freenect::FreenectDevice(ctx, index), m_gamma(2048), m_new_rgb_frame(false) {
        
        m_rgbMat = cv::Mat(cv::Size(640, 480), CV_8UC1, cv::Scalar(0));
        for(auto i = 0; i < 2048; i++) {
            float v = i / 2048.0f;
            v = std::pow(v, 3) * 6;
            m_gamma[i] = v * 6 * 256;
        }
        std::cout << "Created Kinect device!" << std::endl;
    }
    
    void VideoCallback(void* data, uint32_t timestamp) {
        m_rgbMutex.lock();
        uint8_t* rgb = static_cast<uint8_t*>(data);
        m_rgbMat.data = rgb;
        m_new_rgb_frame = true;
        m_rgbMutex.unlock();
    }
    
    bool GetVideo(cv::Mat& output) {
        m_rgbMutex.lock();
        if (m_new_rgb_frame) {
            try {
                cv::cvtColor(m_rgbMat, output, cv::COLOR_GRAY2BGR);
                m_new_rgb_frame = false;
            }
            catch(...) {
                printf("Failed to convert image");
            }
            m_rgbMutex.unlock();
            return true;
        } else {
            m_rgbMutex.unlock();
            return false;
        }
    }
};

std::string getcwd_string( void ) {
    char buff[PATH_MAX];
    getcwd(buff, PATH_MAX);
    string cwd(buff);
    return cwd;
}

std::string filename(std::string extension) {
    std::stringstream ss;
    char buff[20];
    time_t now = time(NULL);
    strftime(buff, 20, "%Y%m%d-%H%M%S", localtime(&now));
    string cwd = getcwd_string();
    
    ss << cwd << "/video/output_" << buff << "." << extension;
    
    return ss.str();
}

int main(int argc, char** argv) {
    cout << "Initializing" << std::endl;
    time_t start = std::clock();
    
    // video writer
    VideoWriter outputVideo;
    Size S = Size(640, 480);
    
    int frames = 0;
    
    int fourcc = VideoWriter::fourcc('H','2','6','4');
    
    string name = filename("mkv");
    cout << name << endl;
    outputVideo.open(name, fourcc, 1, S);
    // end video writer
    
    Freenect::Freenect ctx;
    std::cout << "Created Freenect" << std::endl;
    Kinect& kinect = ctx.createDevice<Kinect>(0);
    std::cout << "Created kinect" << std::endl;
    
    cv::Mat rgb(S, CV_8UC3, Scalar(0));
    
    std::signal(SIGINT, [](int) {
        die = true;
    });
    std::cout << "Set SIGINT handler" << std::endl;
    
    kinect.setVideoFormat(FREENECT_VIDEO_IR_8BIT);
    
    kinect.setFlag(FREENECT_AUTO_EXPOSURE, true);
    kinect.setFlag(FREENECT_AUTO_WHITE_BALANCE, true);
    
    kinect.startVideo();
    
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    
    int gamma_cor = 50;

    float gamma_ = (float)gamma_cor / 100.0;
    
    for( int i = 0; i < 256; ++i) {
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
    }
    
    std::time_t end = std::clock();
    int elapsed = (end - start) / (double) (CLOCKS_PER_SEC / 1000);
    
    std::cout << "Initialization took " << elapsed << " ms"  << std::endl;
    
    while(!die) {
        kinect.updateState();
        if (kinect.GetVideo(rgb)) {
            Mat gamma = rgb.clone();
            LUT(rgb, lookUpTable, gamma);
            outputVideo.write(gamma);
            
            frames++;
            if (frames == 120) {
                frames = 0;
                outputVideo.release();
                std::string name = filename("mkv");
                cout << name << endl;
                outputVideo.open(name, fourcc, 1, S, true);
            }
            usleep(1000000);
        }
    }
    std::cout << "Shutting down..." << std::endl;
    kinect.stopVideo();
    
    kinect.setLed(LED_RED);
}
