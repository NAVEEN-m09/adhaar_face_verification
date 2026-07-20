import os
import requests
import json

def test_api_locally():
    print("=== Aadhaar Verification API Client Test ===")
    url = "http://localhost:8000/verify"
    
    # 1. Create a dummy test image if they don't exist
    # A real test would use a real selfie and real Aadhaar card image
    selfie_path = "selfie_test.jpg"
    aadhaar_path = "aadhaar_test.jpg"
    
    # Generate dummy image bytes if they don't exist
    if not os.path.exists(selfie_path):
        import numpy as np
        import cv2
        # Simple blank blue image as dummy selfie
        dummy_selfie = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw a circle representing a face
        cv2.circle(dummy_selfie, (150, 150), 80, (0, 255, 0), -1)
        cv2.imwrite(selfie_path, dummy_selfie)
        print(f"Generated dummy selfie image at {selfie_path}")

    if not os.path.exists(aadhaar_path):
        import numpy as np
        import cv2
        # Simple blank white image representing card
        dummy_card = np.ones((540, 856, 3), dtype=np.uint8) * 255
        # Draw some lines and shapes to simulate text/card design
        cv2.rectangle(dummy_card, (20, 20), (836, 520), (200, 100, 50), 3)
        # Save card
        cv2.imwrite(aadhaar_path, dummy_card)
        print(f"Generated dummy Aadhaar card image at {aadhaar_path}")

    # Use a dummy Aadhaar number with correct Verhoeff checksum
    aadhaar_num = "3662 1019 8051"
    
    print(f"Sending POST request to {url}...")
    print(f"Payload Aadhaar Number: '{aadhaar_num}'")
    
    # Open files and post
    try:
        with open(selfie_path, 'rb') as f_selfie, open(aadhaar_path, 'rb') as f_card:
            files = {
                'selfie_image': (selfie_path, f_selfie, 'image/jpeg'),
                'aadhaar_image': (aadhaar_path, f_card, 'image/jpeg')
            }
            data = {
                'aadhaar_number': aadhaar_num
            }
            
            response = requests.post(url, files=files, data=data)
            
            print(f"Response Status Code: {response.status_code}")
            print("Response JSON Content:")
            print(json.dumps(response.json(), indent=2))
            
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Connection failed. Make sure the API server is running locally.")
        print("Start it using: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n[ERROR] Test run failed: {str(e)}")
        
    # Cleanup dummy files
    for path in [selfie_path, aadhaar_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

if __name__ == "__main__":
    test_api_locally()
