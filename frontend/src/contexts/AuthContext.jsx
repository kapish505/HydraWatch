import React, { createContext, useContext, useState, useEffect } from 'react';
import { jwtDecode } from 'jwt-decode';
import { API_BASE } from '../config';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(localStorage.getItem('hw_token') || null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Expose fetchUser directly so we can refresh without logging out
  const fetchUser = async (authToken) => {
     try {
         const response = await fetch(`${API_BASE}/api/users/me`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
         });
         if (response.ok) {
            const data = await response.json();
            setUser(data);
         } else {
            // Only force logout if the token is completely invalid
            if (response.status === 401 || response.status === 403) {
               logout();
            }
         }
     } catch (err) {
         console.error("Auth fetch failed", err);
     } finally {
         setLoading(false);
     }
  };

  useEffect(() => {
    if (token) {
      try {
        localStorage.setItem('hw_token', token);
        const decoded = jwtDecode(token);
        
        if (decoded.exp * 1000 < Date.now()) {
          logout();
          setLoading(false);
        } else {
          fetchUser(token);
        }
      } catch (e) {
        logout();
        setLoading(false);
      }
    } else {
      localStorage.removeItem('hw_token');
      setUser(null);
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const login = (newToken, userData) => {
    setToken(newToken);
    setUser(userData);
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('hw_token');
  };

  if (loading && token) {
     return <div className="min-h-screen bg-neutral-950 flex items-center justify-center text-white">Loading Auth...</div>;
  }

  return (
    <AuthContext.Provider value={{ token, user, login, logout, fetchUser }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
