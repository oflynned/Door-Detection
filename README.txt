README

This is a complete Visual Studio Project developed using Visual Studio 2015 and OpenCV 3.1.0 under Windows 10.  It should work on different versions with a little work.

1. You should install OpenCV v3.1.0 on your system in somewhere like C:\   It will automatically create an "opencv" folder in this location.
2. Create an environment variable OPENCV_DIR which should be set to C:\opencv\build\
3. Extend the "Path" system variable by adding ;%OPENCV_DIR%\x64\vc14\bin
4. IF you are using a different version of the OpenCV library you will need to change the names of the included libraries for the Visual Studio project (under Configuration Properties -> Linker -> Input) for both Debug and Release as the library version is specified in the DLL file names.
5. IF the system is failing when started it is likely to be a problem with loading the image and video files (stored in the Media directory).  The simplest solution is to edit the main.cpp file and provide the full path to the media files in the "file_location" variable at the start of the main() routine. 

