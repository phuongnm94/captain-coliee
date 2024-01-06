OPENAI_API_KEY=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

MESSAGE="Hello"


curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": '$MESSAGE'}],
     "temperature": 0.7
   }'
