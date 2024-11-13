const fs = require('fs');
const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const ffmpeg = require('ffmpeg-static');
const multer = require('multer');
const axios = require('axios');

const app = express();
const port = 3000;

const baseDir = __dirname;
const inputDirectory = path.join(baseDir, 'inputvideo');
const outputDirectory = path.join(baseDir, 'outputvideo');
const hls1Directory = path.join(outputDirectory, 'hls_1');
const hls1OutDirectory = path.join(outputDirectory, 'hls1_out');
const hlsUpscaledDirectory = path.join(outputDirectory, 'hls_upscaled');
const publicDirectory = path.join(baseDir, 'public');
const outputFilePath = path.join(hlsUpscaledDirectory, 'final_output.mp4'); // 최종 업스케일된 영상 경로

// 필요한 디렉토리 생성
function createDirectoryIfNotExists(dir) {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
        console.log(`Created directory: ${dir}`);
    }
}

[inputDirectory, outputDirectory, hls1Directory, hls1OutDirectory, hlsUpscaledDirectory, publicDirectory].forEach(createDirectoryIfNotExists);

// 지정된 디렉토리의 모든 파일 삭제
function clearDirectory(directory) {
    fs.readdir(directory, (err, files) => {
        if (err) return console.error(`Error reading directory ${directory}:`, err);
        files.forEach((file) => {
            fs.unlink(path.join(directory, file), (err) => {
                if (err) console.error(`Error deleting file ${file}:`, err);
            });
        });
        console.log(`Cleared all files in directory: ${directory}`);
    });
}

// 서버 시작 시 필요한 디렉토리 정리
clearDirectory(hls1Directory);
clearDirectory(hls1OutDirectory);
clearDirectory(hlsUpscaledDirectory);

// 정적 파일 제공 설정
app.use(express.static(publicDirectory));
app.use('/stream', express.static(hlsUpscaledDirectory));
// outputvideo/hls_upscaled 디렉토리 정적 파일 제공
app.use('/hls_upscaled', express.static(path.join(baseDir, 'outputvideo', 'hls_upscaled')));


// multer 설정 - 업로드된 파일의 원본 이름 사용
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, inputDirectory);
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname);
    }
});

const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'video/mp4') {
            cb(null, true);
        } else {
            cb(new Error('Only MP4 files are allowed'));
        }
    }
});

// HLS 구조에서 프레임을 추출하는 함수
function convertToFramesFromHLS(playlistFile, outputDirectory, prefix) {
    return new Promise((resolve, reject) => {
        const outputPattern = path.join(outputDirectory, `${prefix}_%04d.jpg`);
        const ffmpegProcess = spawn(ffmpeg, [
            '-i', playlistFile,        // HLS .m3u8 파일을 입력으로 사용
            '-vf', 'fps=25',           // 초당 25 프레임 추출
            outputPattern              // 추출된 프레임 저장 경로
        ]);

        ffmpegProcess.on('close', (code) => {
            if (code === 0) {
                console.log(`Frames extracted from ${playlistFile} to ${outputPattern}`);
                resolve();
            } else {
                reject(new Error('Failed to extract frames from HLS'));
            }
        });
    });
}

// HLS 구조에서 오디오를 추출하는 함수
function extractAudioFromHLS(playlistFile, outputDirectory, outputFile) {
    return new Promise((resolve, reject) => {
        const outputFilePath = path.join(outputDirectory, `${outputFile}.aac`);
        const ffmpegProcess = spawn(ffmpeg, [
            '-i', playlistFile,         // HLS .m3u8 파일을 입력으로 사용
            '-q:a', '0',                // 오디오 품질 설정
            '-map', 'a',                // 오디오만 추출
            '-y', outputFilePath        // 추출된 오디오 저장 경로
        ]);

        ffmpegProcess.on('close', (code) => {
            if (code === 0) {
                console.log(`Audio extracted from ${playlistFile} to ${outputFilePath}`);
                resolve(outputFilePath);
            } else {
                reject(new Error('Failed to extract audio from HLS'));
            }
        });
    });
}

// GPU 파드 3개에 알림을 보내는 함수
async function notifyGpuPods(videoFileName) {
    try {
        const gpuServiceUrls = [
            "http://gpu-pod-service:3002/notify1",
            "http://gpu-pod-service:3003/notify2",
            "http://gpu-pod-service:3004/notify3"
        ];

        // 각 GPU 파드에 요청을 비동기적으로 보냄
        const requests = gpuServiceUrls.map(url => {
            return axios.post(url, { status: "processing_complete", videoFileName });
        });

        const responses = await Promise.all(requests);
        responses.forEach(response => {
            console.log("Notification response:", response.data);
        });
        
        return responses;
    } catch (error) {
        console.error("Error in notifyGpuPods function:", error.message);
    }
}

// 비디오를 HLS 세그먼트 및 M3U8 플레이리스트 파일로 변환하는 함수
function convertToHLS(inputFile, outputDirectory, outputPlaylistName) {
    return new Promise((resolve, reject) => {
        const ffmpegProcess = spawn(ffmpeg, [
            '-i', inputFile,
            '-c:v', 'copy',                   // 비디오 스트림을 복사
            '-c:a', 'aac',                     // 오디오 스트림을 AAC로 인코딩
            '-hls_time', '10',                 // 각 HLS 세그먼트의 길이 (초 단위)
            '-hls_list_size', '0',             // 전체 플레이리스트의 세그먼트 수 제한 (0은 모든 세그먼트 포함)
            '-f', 'hls',                       // HLS 형식으로 설정
            path.join(outputDirectory, `${outputPlaylistName}.m3u8`) // HLS 플레이리스트 파일 경로
        ]);

        ffmpegProcess.on('close', (code) => {
            if (code === 0) {
                console.log(`HLS conversion completed for ${inputFile}`);
                resolve();
            } else {
                reject(new Error('Failed to convert video to HLS'));
            }
        });
    });
}

// /upload 엔드포인트 - 파일 업로드 및 HLS 변환
app.post('/upload', upload.single('video'), async (req, res) => {
    try {
        if (!req.file) {
            throw new Error('No file uploaded');
        }

        const videoFileName = req.file.filename;
        const inputFile = path.join(inputDirectory, videoFileName);

        // HLS 형식으로 비디오 변환
        await convertToHLS(inputFile, hls1Directory, 'video_playlist');

        // HLS로 변환된 비디오에서 프레임 및 오디오 추출
        const playlistFile = path.join(hls1Directory, 'video_playlist.m3u8');
        await convertToFramesFromHLS(playlistFile, hls1Directory, 'frame1');
        await extractAudioFromHLS(playlistFile, hls1OutDirectory, 'audio1');

        console.log('Video uploaded, converted to HLS, and frames/audio extracted successfully');

        // GPU 파드에 알림 전송
        await notifyGpuPods(videoFileName);

        res.status(200).json({ 
            status: 'success', 
            message: 'File uploaded, converted to HLS format, frames/audio extracted, and GPU pods notified successfully.',
            hlsPlaylistUrl: `/hls_1/video_playlist.m3u8`
        });
    } catch (error) {
        console.error('Error during upload, HLS conversion, or GPU notification:', error);
        res.status(500).json({ status: 'error', message: error.message });
    }
});


// 각 GPU 파드의 완료 알림 상태를 추적할 변수
let gpuCompletionStatus = {
    gpu1: false,
    gpu2: false,
    gpu3: false
};

// 모든 GPU 파드가 완료되었는지 확인하는 함수
function checkAllGpuCompleted() {
    return gpuCompletionStatus.gpu1 && gpuCompletionStatus.gpu2 && gpuCompletionStatus.gpu3;
}

// /complete1 엔드포인트 - 첫 번째 GPU 파드 완료 알림
app.post('/complete1', async (req, res) => {
    try {
        gpuCompletionStatus.gpu1 = true;
        console.log("Received upscale completion notification from GPU pod 1.");

        // 모든 GPU 파드가 완료되었는지 확인
        if (checkAllGpuCompleted()) {
            await mergeFramesAndAudioToVideo(hls1OutDirectory, 'frame1', 'audio1', outputFilePath);
            console.log(`Final upscaled video saved at ${outputFilePath}`);
        }

        res.status(200).send("Video merging processing completed from GPU pod 1.");
    } catch (error) {
        console.error("Error in /complete1 processing:", error.message);
        res.status(500).json({ status: 'error', message: 'Video merging failed from GPU pod 1' });
    }
});

// /complete2 엔드포인트 - 두 번째 GPU 파드 완료 알림
app.post('/complete2', async (req, res) => {
    try {
        gpuCompletionStatus.gpu2 = true;
        console.log("Received upscale completion notification from GPU pod 2.");

        // 모든 GPU 파드가 완료되었는지 확인
        if (checkAllGpuCompleted()) {
            await mergeFramesAndAudioToVideo(hls1OutDirectory, 'frame1', 'audio1', outputFilePath);
            console.log(`Final upscaled video saved at ${outputFilePath}`);
        }

        res.status(200).send("Video merging processing completed from GPU pod 2.");
    } catch (error) {
        console.error("Error in /complete2 processing:", error.message);
        res.status(500).json({ status: 'error', message: 'Video merging failed from GPU pod 2' });
    }
});

// /complete3 엔드포인트 - 세 번째 GPU 파드 완료 알림
app.post('/complete3', async (req, res) => {
    try {
        gpuCompletionStatus.gpu3 = true;
        console.log("Received upscale completion notification from GPU pod 3.");

        // 모든 GPU 파드가 완료되었는지 확인
        if (checkAllGpuCompleted()) {
            await mergeFramesAndAudioToVideo(hls1OutDirectory, 'frame1', 'audio1', outputFilePath);
            console.log(`Final upscaled video saved at ${outputFilePath}`);
        }

        res.status(200).send("Video merging processing completed from GPU pod 3.");
    } catch (error) {
        console.error("Error in /complete3 processing:", error.message);
        res.status(500).json({ status: 'error', message: 'Video merging failed from GPU pod 3' });
    }
});



// 프레임과 오디오를 결합하여 영상을 생성하는 함수
function mergeFramesAndAudioToVideo(directory, videoPattern, audioFile, outputFile) {
    return new Promise((resolve, reject) => {
        const ffmpegProcess = spawn(ffmpeg, [
            '-i', path.join(directory, `${videoPattern}_%04d_out.jpg`),
            '-i', path.join(directory, `${audioFile}.aac`),
            '-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p', '-shortest', outputFile
        ]);

        ffmpegProcess.on('close', (code) => {
            if (code === 0) {
                console.log(`Video and audio merged successfully to ${outputFile}`);
                resolve();
            } else {
                reject(new Error('Failed to merge video and audio'));
            }
        });
    });
}

// 스트리밍 웹 페이지 제공
app.get('/stream', (req, res) => {
    res.sendFile(path.join(publicDirectory, 'stream.html'));
});

// 서버 시작
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});