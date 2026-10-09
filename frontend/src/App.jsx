import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Predict from './pages/Predict';
import Models from './pages/Models';
import History from './pages/History';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [backendStatus, setBackendStatus] = useState({
    ready: false,
    model_loaded: false,
    best_model_name: '',
    total_features: 0,
    total_diseases: 0
  });

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await axios.get('/api/health');
      if (res.data) {
        setBackendStatus({
          ready: res.data.status === 'healthy',
          model_loaded: res.data.model_loaded,
          best_model_name: res.data.best_model_name,
          total_features: res.data.total_features,
          total_diseases: res.data.total_diseases
        });
      }
    } catch (err) {
      console.warn('Backend /api/health unavailable:', err.message);
      setBackendStatus((prev) => ({ ...prev, ready: false }));
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 font-sans text-slate-800">
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        backendStatus={backendStatus} 
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        {activeTab === 'home' && (
          <Home setActiveTab={setActiveTab} backendStatus={backendStatus} />
        )}
        {activeTab === 'predict' && <Predict />}
        {activeTab === 'models' && <Models />}
        {activeTab === 'history' && <History />}
      </main>

      <Footer />
    </div>
  );
}
