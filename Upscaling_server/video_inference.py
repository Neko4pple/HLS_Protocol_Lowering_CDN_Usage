import cv2
import glob
import math
import numpy as np
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from torch.nn import functional as F
import subprocess

class RealESRGANer():
    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None

        # 모델 초기화
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)

    def pre_process(self, img):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        try:
            # 추론
            with torch.no_grad():
                self.output = self.model(self.img)
        except Exception as error:
            print('Error', error)

    def tile_process(self):
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # 검정 이미지로 시작
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # 모든 타일 반복
        for y in range(tiles_y):
            for x in range(tiles_x):
                # 입력 이미지에서 타일 추출
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # 전체 이미지에서 입력 타일 영역
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # 패딩과 함께 전체 이미지에서 입력 타일 영역
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # 입력 타일 크기
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # 타일 업스케일링
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except Exception as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # 전체 이미지에서 출력 타일 영역
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # 패딩 없는 출력 타일 영역
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # 타일을 출력 이미지에 넣기
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # 추가 패드 제거
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # pre_pad 제거
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output


def main():
    # 입력 및 출력 비디오 파일 정보
    input_video = 'input.mp4'#입력 비디오 경로
    output_video = 'HLS\videos'#출력 비디오 경로
    model_path = 'experiments/pretrained_models/RealESRGAN_x4plus.pth'#모델 가중치 경로
    scale = 4
    tile = 0
    tile_pad = 10
    pre_pad = 0
    alpha_upsampler = 'realesrgan'
    fps = 25

    # 1단계: 입력 비디오에서 프레임 추출
    os.makedirs('inputs', exist_ok=True)
    subprocess.run(['ffmpeg', '-i', input_video, 'inputs/frame_%d.jpg'])#업스케일 전 프레임 저장 경로

    # 2단계: 각 프레임 업스케일링 수행
    upsampler = RealESRGANer(scale=scale, model_path=model_path, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad)
    os.makedirs('results', exist_ok=True)
    paths = sorted(glob.glob(os.path.join('inputs', 'frame_*.jpg')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        max_range = 65535 if np.max(img) > 255 else 255
        img = img / max_range

        if len(img.shape) == 2:  # 회색 이미지
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # 알파 채널이 있는 RGBA 이미지
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        upsampler.pre_process(img)
        if tile:
            upsampler.tile_process()
        else:
            upsampler.process()
        output_img = upsampler.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                upsampler.pre_process(alpha)
                if tile:
                    upsampler.tile_process()
                else:
                    upsampler.process()
                output_alpha = upsampler.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        save_path = f'results/{imgname}.jpg'#업스케일 결과 저장 경로
        if max_range == 65535:
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)
        cv2.imwrite(save_path, output)

    # 3단계: 프레임을 동영상으로 재조립
    subprocess.run(['ffmpeg', '-i', 'results/frame_%d.jpg', '-c:v', 'libx264', '-vf', f'fps={fps}', '-pix_fmt', 'yuv420p', output_video])

    # 4단계: 프레임 삭제
   # for file_path in glob.glob('inputs/*.jpg') + glob.glob('results/*.jpg'):
    #    os.remove(file_path)

if __name__ == '__main__':
    main()
