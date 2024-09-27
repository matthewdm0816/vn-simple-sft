curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"prompt": ["悠馬: 澪と鏡たちは今日くるみたいね。ことしの真紅のたんじょうびですから。", "傍白: 雨は窓外にしずかにおりている。"], "max_length": 300}' \
     | jq -r '.'