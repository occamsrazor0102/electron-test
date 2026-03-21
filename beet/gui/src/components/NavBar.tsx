import { NavLink } from "react-router-dom";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/", label: "Check Text", exact: true },
  { to: "/batch", label: "Batch Analysis" },
  { to: "/history", label: "History" },
  { to: "/help", label: "Help" },
];

export function NavBar() {
  return (
    <header className="sticky top-0 z-40 border-b border-gray-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
        {/* Logo */}
        <NavLink to="/" className="flex items-center gap-2 font-bold text-xl text-gray-900">
          <span
            className="flex h-7 w-7 items-center justify-center rounded-full bg-[#0D7377] text-white text-sm font-black"
            aria-hidden="true"
          >
            B
          </span>
          BEET
        </NavLink>

        {/* Nav links */}
        <nav aria-label="Main navigation" className="flex items-center gap-1">
          {NAV_ITEMS.map(({ to, label, exact }) => (
            <NavLink
              key={to}
              to={to}
              end={exact}
              className={({ isActive }) =>
                clsx(
                  "rounded-lg px-3 py-1.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-[#0D7377]/10 text-[#0D7377]"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                )
              }
            >
              {label}
            </NavLink>
          ))}

          <NavLink
            to="/settings"
            className={({ isActive }) =>
              clsx(
                "ml-2 rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-900 transition-colors",
                isActive && "bg-gray-100 text-gray-900"
              )
            }
            aria-label="Settings"
          >
            ⚙️
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
