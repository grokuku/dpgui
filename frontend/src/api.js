import axios from 'axios';

// On utilise le chemin relatif /api qui sera intercepté par le proxy Vite
// et redirigé vers le backend (ex: http://127.0.0.1:56661)
const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;