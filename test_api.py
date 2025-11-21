import requests

# Endpoint URL
url = "http://127.0.0.1:8000/chat"

# Test 1: Ask a valid question
payload_1 = {
    "session_id": "user_123",
    "query": "How can I reset my password?"
}
response_1 = requests.post(url, json=payload_1)
print("User: How can I reset my password?")
print("Bot:", response_1.json()['response'])

# Test 2: Ask a follow-up (Testing Memory)
payload_2 = {
    "session_id": "user_123", 
    "query": "What did I just ask you?"
}
response_2 = requests.post(url, json=payload_2)
print("\nUser: What did I just ask you?")
print("Bot:", response_2.json()['response'])

# Test 3: Ask an unknown question (Testing Escalation)
payload_3 = {
    "session_id": "user_123",
    "query": "How do I fly a helicopter?"
}
response_3 = requests.post(url, json=payload_3)
print("\nUser: How do I fly a helicopter?")
print("Bot:", response_3.json()['response'])