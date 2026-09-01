"use client";

import { useCallback, useEffect, useState } from "react";
import { PRECONFIGURED_PERSONAS } from "@/constants/personas";
import { OfficerPersona, UserProfile } from "@/types";

const AUTH_TOKEN_KEY = "shasan_auth_token";
const USER_PROFILE_KEY = "shasan_user_profile";

export function useAuth() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [currentPersona, setCurrentPersona] = useState<OfficerPersona>(PRECONFIGURED_PERSONAS[0]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Authenticate with email & password against /api/auth/login
  const login = useCallback(async (email: string, password: string): Promise<UserProfile> => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Authentication failed.");
      }

      const data = await res.json();
      const accessToken = data.access_token as string;
      const userProfile = data.user as UserProfile;

      setToken(accessToken);
      setUser(userProfile);
      localStorage.setItem(AUTH_TOKEN_KEY, accessToken);
      localStorage.setItem(USER_PROFILE_KEY, JSON.stringify(userProfile));

      // Match persona
      const matched = PRECONFIGURED_PERSONAS.find((p) => p.email.toLowerCase() === email.toLowerCase());
      if (matched) {
        setCurrentPersona(matched);
      }

      return userProfile;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // 1-Click Persona Switcher
  const switchPersona = useCallback(async (persona: OfficerPersona) => {
    try {
      setCurrentPersona(persona);
      await login(persona.email, persona.password);
    } catch (err) {
      console.error("Failed to switch officer persona:", err);
    }
  }, [login]);

  // Initial load: restore from localStorage or auto-login default persona
  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem(AUTH_TOKEN_KEY);
      const storedProfile = localStorage.getItem(USER_PROFILE_KEY);

      if (storedToken && storedProfile) {
        try {
          const profile = JSON.parse(storedProfile) as UserProfile;
          setToken(storedToken);
          setUser(profile);

          const matched = PRECONFIGURED_PERSONAS.find((p) => p.email.toLowerCase() === profile.email.toLowerCase());
          if (matched) {
            setCurrentPersona(matched);
          }
          setIsLoading(false);
          return;
        } catch {
          // parse error, fallback
        }
      }

      // Default auto-login as Forest Officer for seamless instant demo
      try {
        await login(PRECONFIGURED_PERSONAS[0].email, PRECONFIGURED_PERSONAS[0].password);
      } catch (err) {
        console.warn("Auto-login default persona failed:", err);
        setIsLoading(false);
      }
    };

    initAuth();
  }, [login]);

  const logout = useCallback(() => {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(USER_PROFILE_KEY);
    setUser(null);
    setToken(null);
  }, []);

  return {
    user,
    token,
    currentPersona,
    isLoading,
    login,
    logout,
    switchPersona,
  };
}
