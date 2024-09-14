#!/bin/bash

wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
tar -xf ffmpeg-git-amd64-static.tar.xz
sudo cp ffmpeg-git-*-amd64-static/ffmpeg /usr/local/bin/
sudo cp ffmpeg-git-*-amd64-static/ffprobe /usr/local/bin/
rm -r ffmpeg-git-*-amd64-static
rm ffmpeg-git-amd64-static.tar.xz
