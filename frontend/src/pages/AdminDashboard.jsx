import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { API_BASE } from '../config';

export default function AdminDashboard() {
  const { token, user: currentUser } = useAuth();
  const [users, setUsers] = useState([]);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchData = async () => {
    try {
      const [usersRes, logsRes] = await Promise.all([
        fetch(`${API_BASE}/api/admin/users`, { headers: { 'Authorization': `Bearer ${token}` } }),
        fetch(`${API_BASE}/api/admin/activity`, { headers: { 'Authorization': `Bearer ${token}` } })
      ]);
      const usersData = await usersRes.json();
      const logsData = await logsRes.json();
      setUsers(usersData.users || []);
      setLogs(logsData.logs || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const toggleStatus = async (user) => {
     if (user.id === currentUser.id) return alert("Cannot alter your own status.");
     const newStatus = user.status === 'active' ? 'suspended' : 'active';
     try {
        const res = await fetch(`${API_BASE}/api/admin/users/${user.id}/status`, {
           method: 'PUT',
           headers: { 
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json'
           },
           body: JSON.stringify({ status: newStatus })
        });
        if (res.ok) fetchData();
     } catch (err) {
        console.error("Status update error", err);
     }
  };

  if (loading) return <div className="p-8 text-neutral-400 font-medium">Booting Administrative Console...</div>;

  return (
    <div className="p-8 max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8 animate-in fade-in duration-500">
      
      <div className="lg:col-span-2 space-y-4">
         <h2 className="text-xl font-bold tracking-wider text-white mb-4 flex items-center space-x-2">
            <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse mr-2" />
            Operator Clearance Roster
         </h2>
         <div className="hw-panel border border-white/5 rounded-xl overflow-hidden shadow-2xl">
            <table className="w-full text-left">
               <thead className="bg-black/40 border-b border-white/5 text-xs uppercase tracking-wider text-neutral-500">
                  <tr>
                     <th className="px-4 py-3">ID</th>
                     <th className="px-4 py-3">Operator</th>
                     <th className="px-4 py-3">Role</th>
                     <th className="px-4 py-3">Status</th>
                     <th className="px-4 py-3 text-right">Overrides</th>
                  </tr>
               </thead>
               <tbody className="divide-y divide-white/5">
                  {users.map(u => (
                     <tr key={u.id} className="hover:bg-white/5 transition-colors group">
                        <td className="px-4 py-3 text-neutral-400 text-sm">#{u.id}</td>
                        <td className="px-4 py-3 text-white text-sm flex items-center space-x-3">
                           {u.profile_picture_url ? (
                              <img src={`${API_BASE}${u.profile_picture_url}`} className="w-8 h-8 rounded-full object-cover border border-white/10" />
                           ) : (
                              <div className="w-8 h-8 rounded-full bg-blue-900/30 flex items-center justify-center border border-white/10 font-bold">{u.email.charAt(0).toUpperCase()}</div>
                           )}
                           <span className="font-medium">{u.email}</span>
                           {u.id === currentUser.id && <span className="text-[10px] bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded ml-2">YOU</span>}
                        </td>
                        <td className="px-4 py-3 text-sm text-neutral-400">{u.role}</td>
                        <td className="px-4 py-3">
                           <span className={`inline-block px-2 py-0.5 rounded text-[11px] font-medium tracking-wide uppercase shadow-inner ${u.status === 'active' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
                              {u.status}
                           </span>
                        </td>
                        <td className="px-4 py-3 text-right">
                           {u.id !== currentUser.id && (
                              <button 
                                 onClick={() => toggleStatus(u)}
                                 className="text-xs text-neutral-400 hover:text-white px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 transition-colors opacity-0 group-hover:opacity-100"
                              >
                                 {u.status === 'active' ? 'Revoke Access' : 'Reactivate'}
                              </button>
                           )}
                        </td>
                     </tr>
                  ))}
               </tbody>
            </table>
         </div>
      </div>
      
      <div className="space-y-4">
         <h2 className="text-xl font-bold tracking-wider text-white mb-4">System Event Log</h2>
         <div className="hw-panel border border-white/5 rounded-xl overflow-hidden p-0 max-h-[600px] overflow-y-auto custom-scrollbar">
            <ul className="divide-y divide-white/5">
               {logs.map(log => (
                  <li key={log.id} className="px-4 py-3 hover:bg-white/5 transition-colors">
                     <div className="flex justify-between items-start mb-1">
                        <span className="font-medium text-blue-400 text-sm truncate max-w-[150px]">{log.email}</span>
                        <span className="text-[11px] text-neutral-500 whitespace-nowrap ml-2 bg-black/50 px-1.5 rounded border border-white/5">
                           {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                     </div>
                     <div className="text-neutral-300 bg-white/10 font-mono text-[11px] rounded px-2 py-1 inline-block mt-1 shadow-inner border border-white/5">
                        {log.action}
                     </div>
                  </li>
               ))}
               {logs.length === 0 && <div className="p-4 text-center text-sm text-neutral-500">No events logged</div>}
            </ul>
         </div>
      </div>
      
    </div>
  );
}
