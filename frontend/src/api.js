import axios from 'axios';

// On tape sur le chemin relatif '/api'.
// Le proxy de Vite (configuré ci-dessus) redirigera vers le bon port backend.
const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;