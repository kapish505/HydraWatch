import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { API_BASE } from '../config';

export default function Profile() {
  const { user, token, fetchUser } = useAuth();
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setMessage('');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/users/me/picture`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });

      if (!res.ok) throw new Error('Upload failed');
      
      setMessage('Profile updated successfully');
      setFile(null);
      if (fetchUser) fetchUser(token);
      
    } catch (err) {
      setMessage(`Error: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-8 max-w-2xl mx-auto space-y-8 animate-in fade-in duration-500">
      <h2 className="text-2xl font-bold tracking-wider text-white mb-6">Operator Profile</h2>
      
      <div className="hw-panel p-6 rounded-2xl border border-white/5 space-y-6">
         <div className="flex items-start space-x-6">
            <div className="flex-shrink-0">
               {user.profile_picture_url ? (
                  <img src={`${API_BASE}${user.profile_picture_url}`} alt="Avatar" className="w-24 h-24 rounded-full border border-white/20 object-cover shadow-lg" />
               ) : (
                  <div className="w-24 h-24 rounded-full bg-blue-900/30 flex items-center justify-center border border-blue-500/30 text-3xl font-bold text-white">
                     {user.email.charAt(0).toUpperCase()}
                  </div>
               )}
            </div>
            
            <div className="space-y-4 flex-1">
               <div>
                  <div className="text-xs uppercase tracking-wider text-neutral-500">Identity</div>
                  <div className="text-lg text-white font-medium">{user.email}</div>
               </div>
               <div className="flex space-x-4">
                  <div>
                     <div className="text-xs uppercase tracking-wider text-neutral-500">Clearance</div>
                     <div className="inline-block px-2 py-0.5 rounded text-xs mt-1 bg-white/10 text-neutral-300 border border-white/10 shadow-inner">
                        {user.role}
                     </div>
                  </div>
                  <div>
                     <div className="text-xs uppercase tracking-wider text-neutral-500">Status</div>
                     <div className="inline-block px-2 py-0.5 rounded text-xs mt-1 bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-inner">
                        {user.status}
                     </div>
                  </div>
               </div>
            </div>
         </div>
         
         <div className="pt-6 border-t border-white/5">
            <h3 className="text-sm font-medium text-neutral-300 mb-4">Update Identifier</h3>
            <form onSubmit={handleUpload} className="flex items-center space-x-4">
               <input 
                  type="file" 
                  accept="image/*"
                  onChange={(e) => setFile(e.target.files[0])}
                  className="text-sm text-neutral-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-white/10 file:text-white hover:file:bg-white/20 cursor-pointer"
               />
               <button 
                  type="submit" 
                  disabled={!file || uploading}
                  className="px-4 py-2 bg-blue-600/80 hover:bg-blue-500 text-white rounded-md text-sm transition-colors disabled:opacity-50"
               >
                  {uploading ? 'Processing...' : 'Upload'}
               </button>
            </form>
            {message && <div className="mt-4 text-sm text-neutral-400">{message}</div>}
         </div>
      </div>
    </div>
  );
}
