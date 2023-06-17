export http_proxy=http://100.72.64.19:12798 && export https_proxy=http://100.72.64.19:12798
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
unset http_proxy && unset https_proxy