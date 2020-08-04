// based on: https://gist.github.com/DrWateryCat/c225e436d5761bbb21fd622c78e78972
#define DEBUG

#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>

#include <libusb-1.0/libusb.h>
#include <libfreenect/libfreenect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <pthread.h>

#include <csignal>

using namespace std;
using namespace cv;
//using namespace cs;

static bool USE_REGISTERED = true;
static volatile bool die = false;
std::string text = "";

cv::Point depthPoint(320, 240);

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
    std::vector<uint8_t> m_depth_buffer;
    std::vector<uint8_t> m_color_buffer;
    std::vector<uint16_t> m_gamma;
    
    cv::Mat m_depthMat;
    cv::Mat m_rgbMat;
    cv::Mat m_ownMat;
    
    myMutex m_depthMutex;
    myMutex m_rgbMutex;
    
    bool m_new_depth_frame = false;
    bool m_new_rgb_frame = false;
    
public:
    Kinect(freenect_context* ctx, int index)
    :Freenect::FreenectDevice(ctx, index) {
        //std::cout << "Inside Constructor" << std::endl;
        m_new_depth_frame = false;
        m_new_rgb_frame = false;
        
        m_rgbMat = cv::Mat(cv::Size(640, 480), CV_8UC3, cv::Scalar(0));
        m_depthMat = cv::Mat(cv::Size(640, 480), CV_16UC1);
        m_gamma = std::vector<uint16_t>(2048);
        for(auto i = 0; i < 2048; i++) {
            float v = i / 2048.0f;
            v = std::pow(v, 3) * 6;
            m_gamma[i] = v * 6 * 256;
        }
        
        //this->setDepthFormat((freenect_depth_FREENECT_DEPTH_11BIT | FREENECT_DEPTH_MM, FREENECT_RESOLUTION_MEDIUM);
        //std::cout << "Created Kinect device!" << std::endl;
    }
    
    void VideoCallback(void* data, uint32_t timestamp) {
        m_rgbMutex.lock();
        uint8_t* rgb = static_cast<uint8_t*>(data);
        m_rgbMat.data = rgb;
        m_new_rgb_frame = true;
        m_rgbMutex.unlock();
    }
    
    void DepthCallback(void* data, uint32_t timestamp) {
        m_depthMutex.lock();
        //uint16_t* depth = static_cast<uint16_t*>(data);
        m_depthMat.data = static_cast<uint8_t*>(data);
        m_new_depth_frame = true;
        m_depthMutex.unlock();
    }
    
    bool GetVideo(cv::Mat& output) {
        m_rgbMutex.lock();
        if (m_new_rgb_frame) {
            cv::cvtColor(m_rgbMat, output, cv::COLOR_RGB2BGR);
            m_new_rgb_frame = false;
            m_rgbMutex.unlock();
            return true;
        } else {
            m_rgbMutex.unlock();
            return false;
        }
    }
    
    bool GetDepth(cv::Mat& output) {
        m_depthMutex.lock();
        if(m_new_depth_frame) {
            m_depthMat.copyTo(output);
            m_new_depth_frame = false;
            m_depthMutex.unlock();
            return true;
        } else {
            m_depthMutex.unlock();
            return false;
        }
    }
};

cv::Mat convertRawDepthToColor(cv::Mat depth) {
    cv::Mat ret = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
    uchar* depth_mid = ret.data;
    
    int i;
    for (i = 0; i < 640*480; i++) {
        int lb = ((short *)depth.data)[i] % 256;
        int ub = ((short *)depth.data)[i] / 256;
        switch (ub) {
            case 0:
                depth_mid[3*i+2] = 255;
                depth_mid[3*i+1] = 255-lb;
                depth_mid[3*i+0] = 255-lb;
                break;
            case 1:
                depth_mid[3*i+2] = 255;
                depth_mid[3*i+1] = lb;
                depth_mid[3*i+0] = 0;
                break;
            case 2:
                depth_mid[3*i+2] = 255-lb;
                depth_mid[3*i+1] = 255;
                depth_mid[3*i+0] = 0;
                break;
            case 3:
                depth_mid[3*i+2] = 0;
                depth_mid[3*i+1] = 255;
                depth_mid[3*i+0] = lb;
                break;
            case 4:
                depth_mid[3*i+2] = 0;
                depth_mid[3*i+1] = 255-lb;
                depth_mid[3*i+0] = 255;
                break;
            case 5:
                depth_mid[3*i+2] = 0;
                depth_mid[3*i+1] = 0;
                depth_mid[3*i+0] = 255-lb;
                break;
            default:
                depth_mid[3*i+2] = 0;
                depth_mid[3*i+1] = 0;
                depth_mid[3*i+0] = 0;
                break;
        }
    }
    return ret;
}

cv::Mat RegisteredDepthToColor(cv::Mat depth) {
    cv::Mat ret;
    
    double min;
    double max;
    
    cv::minMaxIdx(depth, &min, &max, 0, 0);
    
    cv::Mat adjMap;
    depth.convertTo(adjMap, CV_8UC1, 255 / (max - min), -min);
    
    cv::applyColorMap(adjMap, ret, cv::COLORMAP_OCEAN);
    
    return ret;
}

cv::Mat ConvertDepthToColor(cv::Mat depth) {
    if (USE_REGISTERED) {
        return RegisteredDepthToColor(depth);
    } else {
        return convertRawDepthToColor(depth);
    }
}

float RawDepthToMeters(uint16_t depthData)
{
    if (depthData < 2047) {
        return 1.0 / (depthData * -0.0030711016 + 3.3309495161);
    }
    return 0;
}

float RegisteredDepthToMeters(uint16_t depthData) {
    return (float)depthData / 1000;
}

float DistanceTo(cv::Point2i point, cv::Mat depth) {
    if (USE_REGISTERED) {
        return RegisteredDepthToMeters(depth.at<uint16_t>(point));
    } else {
        return RawDepthToMeters(depth.at<uint16_t>(point));
    }
}

//void MouseCallbackFunc(int event, int x, int y, int flags, void* userData) {
//    if (event == CV_EVENT_LBUTTONDOWN) {
//        depthPoint.x = x;
//        depthPoint.y = y;
//
//        //std::cout << "Got mouse click at X: " << x << ", Y: " << y << std::endl;
//    }
//    if (event == CV_EVENT_RBUTTONDOWN) {
//        depthPoint.x = 320;
//        depthPoint.y = 240;
//
//        //std::cout << "Reset point" << std::endl;
//    }
//}

void ParseCommandLine(int argc, char** argv) {
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            std::string arg = std::string(argv[i]);
            if (arg == "--use-registered") {
                USE_REGISTERED = true;
            }
        }
    } else {
        USE_REGISTERED = false;
    }
}

int main(int argc, char** argv) {
    std::cout << "Initializing" << std::endl;
    std::time_t start = std::clock();
    
    ParseCommandLine(argc, argv);
    
    Freenect::Freenect ctx;
    //std::cout << "Created Freenect" << std::endl;
    Kinect& kinect = ctx.createDevice<Kinect>(0);
    //std::cout << "Created kinect" << std::endl;
    
//    cs::MjpegServer rgbServer{"rgb", 1185};
//    cs::MjpegServer depthServer{"depth", 1186};
    //std::cout << "Created mjpg servers" << std::endl;
    
//    cs::CvSource rgbSource{"rgbSource", cs::VideoMode::kMJPEG, 640, 480, 30};
//    cs::CvSource depthSource{"depthSource", cs::VideoMode::kBGR, 640, 480, 30};
    
//    rgbServer.SetSource(rgbSource);
//    depthServer.SetSource(depthSource);
    //std::cout << "Created and set CvSources" << std::endl;
    
//    cs::SetDefaultLogger(1);
    
    cv::Mat rgb(Size(640, 480), CV_8UC3, Scalar(0));
//    cv::Mat depth(Size(640, 480), CV_16UC1);
//    cv::Mat depth_viewable = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
    
    std::signal(SIGINT, [](int) {
        die = true;
    });
    //std::cout << "Set SIGINT handler" << std::endl;
    
//    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
//    cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
    
//    cv::setMouseCallback("Video", MouseCallbackFunc, NULL);
//    cv::setMouseCallback("Depth", MouseCallbackFunc, NULL);
    
//    if (USE_REGISTERED) {
//        kinect.setDepthFormat(FREENECT_DEPTH_REGISTERED, FREENECT_RESOLUTION_MEDIUM);
//    } else {
//        kinect.setDepthFormat(FREENECT_DEPTH_11BIT, FREENECT_RESOLUTION_MEDIUM);
//    }
    
    kinect.setFlag(FREENECT_AUTO_EXPOSURE, true);
    //kinect.setFlag(FREENECT_AUTO_WHITE_BALANCE, true);
    
    kinect.startVideo();
//    kinect.startDepth();
    
    //std::cout << "Started depth and video streams" << std::endl;
    
    kinect.setLed(LED_GREEN);
//    kinect.setTiltDegrees(0);
    
    std::time_t end = std::clock();
    int elapsed = (end - start) / (double) (CLOCKS_PER_SEC / 1000);
    
    std::cout << "Initialization took " << elapsed << " ms"  << std::endl;
    
    while(!die) {
        //std::cout << "Getting frame" << std::endl;
        if (!kinect.GetVideo(rgb)) {
            rgb = cv::Mat(640, 480, CV_8UC3, cv::Scalar(255, 0, 0));
            std::cout << "Failed to get frame from kinect" << std::endl;
        }
        
//        if (!kinect.GetDepth(depth)) {
//            depth = cv::Mat(640, 480, CV_16UC1, cv::Scalar(0));
//            std::cout << "Failed to get frame from kinect" << std::endl;
//        }
        
//        float distance = DistanceTo(depthPoint, depth);
        //std::cout << "Distance: " << distance << " Raw: " << (int)(depth.at<uint16_t>(depthPoint)) << '\r' << std::flush;
        
//        cv::circle(rgb, depthPoint, 5, cv::Scalar(255, 255, 255), 1);
//        depth_viewable = ConvertDepthToColor(depth);
//        cv::circle(depth_viewable, depthPoint, 5, cv::Scalar(255, 255, 255), 1);
//        std::ostringstream ss;
//        ss << "Distance: " << distance;
//        cv::putText(depth_viewable, ss.str(), cv::Point(320, 440), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
//        cv::putText(rgb, ss.str(), cv::Point(320, 440), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        
//        if (TRY_CSCORE) {
//            rgbSource.PutFrame(rgb);
//            //depthSource.PutFrame(depth_viewable);
//        }
        
    }
    std::cout << "Shutting down..." << std::endl;
//    kinect.stopDepth();
    kinect.stopVideo();
    
    kinect.setLed(LED_RED);
}
