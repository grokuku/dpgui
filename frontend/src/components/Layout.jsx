import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { LayoutDashboard, List, Database, Settings } from 'lucide-react';

const Layout = () => {
  return (
    <div className="app-layout">
      <aside className="sidebar">
        <h1>DPGui Manager</h1>
        <nav>
          <NavLink to="/" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            <LayoutDashboard size={20} />
            <span>Dashboard</span>
          </NavLink>
          
          <NavLink to="/jobs" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            <List size={20} />
            <span>Jobs Manager</span>
          </NavLink>

          <NavLink to="/datasets" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            <Database size={20} />
            <span>Datasets</span>
          </NavLink>

          <NavLink to="/settings" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            <Settings size={20} />
            <span>Settings</span>
          </NavLink>
        </nav>
      </aside>

      <main className="main-content-area">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;