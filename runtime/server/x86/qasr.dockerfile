FROM mobvoiwenet/wenet:v0.1.0

RUN rm -rf /home/wenet/
COPY ./wenet/ /home/wenet/
WORKDIR /home/wenet/runtime/server/x86/build/
RUN cmake ..
RUN cmake --build .