/**
 * Central API configuration for HydraWatch frontend.
 *
 * In development, Vite injects VITE_API_URL from .env or defaults to localhost.
 * In production, set VITE_API_URL at build time to point to the real backend.
 */

export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// WebSocket base — derives from API_BASE automatically
const wsProto = API_BASE.startsWith('https') ? 'wss' : 'ws';
export const WS_BASE = `${wsProto}://${new URL(API_BASE).host}`;
