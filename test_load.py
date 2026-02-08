import requests
import threading
import time
import json
import random

# Configuration
url = "http://127.0.0.1:5000/predict"
concurrent_users = 10
total_requests = 100

def simulate_user(user_id):
    # Random sample input
    data = {
        "age": random.randint(20, 80),
        "bmi": round(random.uniform(18.5, 35.0), 1),
        "bp": random.randint(100, 160),
        "cholesterol": random.randint(150, 280),
        "glucose": random.randint(70, 180),
        "gender": random.randint(0, 1)
    }
    
    try:
        start = time.time()
        response = requests.post(url, json=data)
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            res_json = response.json()
            print(f"User {user_id}: OK | Client latency: {elapsed:.1f}ms | API latency: {res_json.get('latency_ms', 'N/A')}")
        else:
            print(f"User {user_id}: Request failed ({response.status_code})")
            
    except Exception as e:
        print(f"User {user_id}: Error - {str(e)}")

print(f"Starting load test with {concurrent_users} concurrent users...")
threads = []

start_test = time.time()

for i in range(total_requests):
    t = threading.Thread(target=simulate_user, args=(i,))
    threads.append(t)
    t.start()
    
    # Simple limiter: keep roughly concurrent_users in flight
    # A thread pool would be more precise, but this keeps it lightweight.
    if len(threads) >= concurrent_users:
        for t in threads:
            t.join()
        threads = []

# Wait for remaining
for t in threads:
    t.join()

end_test = time.time()
print(f"\nTest completed.")
print(f"Total Time: {end_test - start_test:.2f}s")
print(f"Requests per second: {total_requests / (end_test - start_test):.2f}")
