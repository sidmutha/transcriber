all: transcribe run

transcribe: Transcriber.cpp
	g++ -g -O2 Transcriber.cpp -lm -o transcribe `pkg-config --cflags --libs opencv`

run: transcribe
	./transcribe ./videos/rem.mp4 4
