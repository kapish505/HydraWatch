import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { Droplet, LayoutDashboard, ShieldCheck, LogOut, User } from 'lucide-react';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import Profile from './pages/Profile';
import AdminDashboard from './pages/AdminDashboard';

const ProtectedRoute = ({ children, adminOnly = false }) => {
  const { user, token } = useAuth();
  
  if (!token || !user) return <Navigate to="/login" replace />;
  if (adminOnly && user?.role !== 'admin') return <Navigate to="/" replace />;
  
  return children;
};

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
         <div className="min-h-screen bg-background text-on-surface font-['Inter'] flex flex-col">
           <Routes>
             <Route path="/" element={<Home />} />
             <Route path="/login" element={<Login />} />
             <Route 
               path="/dashboard" 
               element={
                 <ProtectedRoute>
                   <Dashboard />
                 </ProtectedRoute>
               } 
             />
             <Route 
               path="/profile" 
               element={
                 <ProtectedRoute>
                   <Profile />
                 </ProtectedRoute>
               } 
             />
             <Route 
               path="/admin" 
               element={
                  <ProtectedRoute adminOnly={true}>
                    <AdminDashboard />
                  </ProtectedRoute>
               } 
             />
             <Route path="*" element={<Navigate to="/" replace />} />
           </Routes>
         </div>
      </BrowserRouter>
    </AuthProvider>
  );
}
