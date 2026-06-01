cmake_minimum_required(VERSION 3.10)
project(ObjectTracker)

set(CMAKE_CXX_STANDARD 17)

# Office OpenCV 4.13.0 path
set(OpenCV_DIR "C:/OpenCV/x64/vc17/lib")
find_package(OpenCV REQUIRED)

if(MSVC)
    add_compile_options(/W3)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ObjectTracker
    main.cpp
    KalmanTracker.cpp
    Detector.cpp
    Tracker.cpp
    FrameStabilizer.cpp
    ThreadedCamera.cpp
)

target_link_libraries(ObjectTracker
    ${OpenCV_LIBS}
    optimized opencv_tracking4130
    debug opencv_tracking4130d
)

if(WIN32)
    add_custom_command(TARGET ObjectTracker
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        -E copy_directory
        "C:/OpenCV/x64/vc17/bin"
        $<TARGET_FILE_DIR:ObjectTracker>)
endif()
