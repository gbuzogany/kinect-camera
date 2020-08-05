CC=g++
CFLAGS=-I/usr/include/libusb-1.0 -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_core -lfreenect -lpthread
OBJ = main.cpp

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

kinect-camera: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)