cmake_minimum_required(VERSION 3.10)
project(ObjectTracker)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/opencv_build_new/install/lib")
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
)

if(WIN32)
    add_custom_command(TARGET ObjectTracker
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        -E copy_directory
        "C:/opencv_build_new/install/bin"
        $<TARGET_FILE_DIR:ObjectTracker>)
endif()
