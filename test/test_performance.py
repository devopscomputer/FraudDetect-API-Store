import requests
import time

class TestAPIPerformance:
    
    def test_response_time(self):
        start_time = time.time()
        response = requests.post('http://localhost:8000/prediction', json={"data": [0.1, 0.2, 0.3]})
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1  # Espera-se que a resposta seja retornada em menos de 1 segundo