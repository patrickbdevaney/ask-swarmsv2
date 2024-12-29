import requests

# Define the base URL for the FastAPI server
BASE_URL = "http://127.0.0.1:8000"

def chat_with_api(user_input):
    """Send a chat request to the /chat endpoint."""
    url = f"{BASE_URL}/chat"
    response = requests.post(url, json={"input_text": user_input})
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}, {response.text}"

def main():
    print("Welcome to the automated chatbot interface!")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() == "quit":
            break
        
        # Automatically send the user input to the /chat endpoint
        print("Chatbot is thinking...")
        response = chat_with_api(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
