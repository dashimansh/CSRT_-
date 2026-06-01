#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>

class ThreadedCamera
{
public:
    ThreadedCamera();
    ~ThreadedCamera();

    bool Open(int DeviceID = 0);
    bool OpenFile(
        const std::string& Path);
    bool GetLatestFrame(cv::Mat& Out);
    void Stop();

    bool IsRunning() const
    {
        return bRunning.load();
    }
    int GetWidth()  const { return Width;  }
    int GetHeight() const { return Height; }
    int GetFPS()    const { return FPS;    }

private:
    cv::VideoCapture  Cap;
    cv::Mat           LatestFrame;
    std::mutex        FrameMutex;
    std::thread       CamThread;
    std::atomic<bool> bRunning{ false };
    std::atomic<bool> bNewFrame{ false };
    int Width = 0, Height = 0, FPS = 0;

    void CaptureLoop();
    void SetupCamera();
};
