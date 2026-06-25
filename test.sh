curl -X POST http://localhost:19600/pre_svc/ -H 'Content-Type: application/json' -d '{"vid":"2","file":"2.mp4"}'
curl -X POST http://localhost:19600/stt_svc/ -H 'Content-Type: application/json' -d '{"vid":"2","file_path":"output/2/audio.wav"}'

