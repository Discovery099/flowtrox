{
  "product": {
    "name": "FLOWTOX_REGIME_01 Dashboard",
    "design_personality": [
      "Bloomberg-terminal quant aesthetic",
      "dense, data-rich, precise",
      "institutional / research-grade (honest failure states)",
      "dark-first with restrained chroma",
      "monospaced numerics + crisp labels"
    ],
    "north_star": "Make it feel like a serious trading desk tool: fast scanning, compact grids, clear P&L semantics, and a terminal-like progress console for long jobs."
  },

  "layout": {
    "information_architecture": {
      "primary_page": "Single dashboard",
      "navigation": {
        "pattern": "Persistent left control rail + top status strip; main content uses tabs",
        "tabs": [
          "Overview (Scorecard + Metrics + Equity)",
          "Charts (Regime/Vol/Distribution/Sensitivity)",
          "Trades (table + filters + downloads)",
          "Model (HAR/GMM/thresholds/best params)",
          "About (nice-to-have)"
        ]
      }
    },
    "grid_system": {
      "container": "max-w-none w-full",
      "desktop": "12-col CSS grid; left rail fixed 320px; main area fluid",
      "tablet": "left rail collapses into Sheet; main becomes single column with section stacking",
      "mobile": "top sticky run CTA + Sheet for parameters; tabs become horizontal scroll"
    },
    "recommended_page_skeleton": {
      "top_bar": "Instrument, last run timestamp, job status pill, quick download menu",
      "left_rail": "Run Panel + presets + run buttons + progress summary",
      "main": "Tabs -> each tab is a dense grid of Cards with compact headers"
    }
  },

  "typography": {
    "font_pairing": {
      "ui_sans": {
        "name": "IBM Plex Sans",
        "usage": "labels, headings, UI chrome"
      },
      "mono": {
        "name": "IBM Plex Mono",
        "usage": "all numerics, tickers, metrics, tables, console"
      }
    },
    "import_notes": {
      "google_fonts": [
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
      ],
      "css_application": {
        "body": "font-family: var(--font-sans)",
        "numeric": "font-variant-numeric: tabular-nums; font-family: var(--font-mono)"
      }
    },
    "type_scale_tailwind": {
      "h1": "text-4xl sm:text-5xl lg:text-6xl font-semibold tracking-tight",
      "h2": "text-base md:text-lg font-medium text-muted-foreground",
      "section_title": "text-sm font-semibold tracking-wide uppercase",
      "kpi_value": "text-xl md:text-2xl font-semibold",
      "table": "text-xs md:text-sm",
      "mono_numbers": "font-mono tabular-nums"
    }
  },

  "color_system": {
    "mode": "dark-first (no light theme required)",
    "semantic_intent": "Green/red reserved for meaning (P&L, pass/fail). Avoid decorative gradients.",
    "tokens_hsl_for_shadcn": {
      "background": "220 18% 6%",
      "foreground": "210 20% 96%",
      "card": "220 18% 8%",
      "card-foreground": "210 20% 96%",
      "popover": "220 18% 8%",
      "popover-foreground": "210 20% 96%",
      "primary": "210 20% 96%",
      "primary-foreground": "220 18% 8%",
      "secondary": "220 14% 14%",
      "secondary-foreground": "210 20% 96%",
      "muted": "220 14% 12%",
      "muted-foreground": "215 12% 70%",
      "accent": "195 85% 45%",
      "accent-foreground": "220 18% 8%",
      "destructive": "0 72% 52%",
      "destructive-foreground": "210 20% 96%",
      "border": "220 14% 18%",
      "input": "220 14% 18%",
      "ring": "195 85% 45%",
      "radius": "0.6rem",
      "chart-1": "145 70% 45%",
      "chart-2": "0 72% 52%",
      "chart-3": "195 85% 45%",
      "chart-4": "45 90% 55%",
      "chart-5": "215 12% 70%"
    },
    "additional_semantic_tokens_css": {
      "--pos": "145 70% 45%",
      "--neg": "0 72% 52%",
      "--warn": "45 90% 55%",
      "--info": "195 85% 45%",
      "--surface-0": "220 18% 6%",
      "--surface-1": "220 18% 8%",
      "--surface-2": "220 14% 12%",
      "--hairline": "220 14% 18%"
    },
    "usage_rules": {
      "pnl": {
        "positive": "text-[hsl(var(--pos))]",
        "negative": "text-[hsl(var(--neg))]",
        "neutral": "text-muted-foreground"
      },
      "pass_fail": {
        "pass_chip": "bg-[hsl(var(--pos)/0.12)] text-[hsl(var(--pos))] border-[hsl(var(--pos)/0.35)]",
        "fail_chip": "bg-[hsl(var(--neg)/0.12)] text-[hsl(var(--neg))] border-[hsl(var(--neg)/0.35)]",
        "warn_chip": "bg-[hsl(var(--warn)/0.12)] text-[hsl(var(--warn))] border-[hsl(var(--warn)/0.35)]"
      }
    },
    "texture": {
      "noise_overlay": "Use a subtle CSS noise overlay (opacity 0.035–0.06) on the app background only; never on cards/tables.",
      "grid_overlay": "Optional 1px grid overlay at 4rem spacing with very low alpha (0.04) to evoke terminal plotting paper."
    },
    "gradients": {
      "policy": "Avoid gradients except tiny decorative overlays (<20% viewport). No saturated purple/pink combos.",
      "allowed_example": "radial-gradient(600px circle at 20% 0%, hsla(195,85%,45%,0.10), transparent 55%), radial-gradient(700px circle at 80% 10%, hsla(145,70%,45%,0.08), transparent 60%)"
    }
  },

  "components": {
    "shadcn_component_paths": {
      "buttons": "/app/frontend/src/components/ui/button.jsx",
      "inputs": "/app/frontend/src/components/ui/input.jsx",
      "slider": "/app/frontend/src/components/ui/slider.jsx",
      "select": "/app/frontend/src/components/ui/select.jsx",
      "tabs": "/app/frontend/src/components/ui/tabs.jsx",
      "card": "/app/frontend/src/components/ui/card.jsx",
      "badge": "/app/frontend/src/components/ui/badge.jsx",
      "progress": "/app/frontend/src/components/ui/progress.jsx",
      "scroll_area": "/app/frontend/src/components/ui/scroll-area.jsx",
      "table": "/app/frontend/src/components/ui/table.jsx",
      "separator": "/app/frontend/src/components/ui/separator.jsx",
      "tooltip": "/app/frontend/src/components/ui/tooltip.jsx",
      "dialog": "/app/frontend/src/components/ui/dialog.jsx",
      "sheet": "/app/frontend/src/components/ui/sheet.jsx",
      "sonner_toast": "/app/frontend/src/components/ui/sonner.jsx"
    },
    "custom_components_to_create": {
      "TerminalShell": {
        "purpose": "Reusable terminal-like panel wrapper for console + dense readouts",
        "notes": "Use Card with inset border + mono font + scanline/noise overlay"
      },
      "MetricKPI": {
        "purpose": "Compact KPI tile with label, value, delta, and optional sparkline",
        "notes": "Value in mono; color by semantic intent"
      },
      "AcceptanceChip": {
        "purpose": "PASS/FAIL chip with criterion label and threshold",
        "notes": "Use Badge variant + semantic colors"
      },
      "RunStatusPill": {
        "purpose": "Top bar job status (IDLE/RUNNING/DONE/FAILED) with spinner",
        "notes": "Use Badge + Progress"
      },
      "DenseDataTable": {
        "purpose": "Trade log table with sticky header, column pinning (optional), sorting/filtering",
        "notes": "Use shadcn Table + ScrollArea; keep row height compact"
      },
      "ChartCard": {
        "purpose": "Standard chart container with header controls (download, toggle series)",
        "notes": "Use Card + Tabs/ToggleGroup"
      }
    },
    "button_system": {
      "tone": "Professional / Corporate",
      "radius": "rounded-md (6–8px)",
      "variants": {
        "primary": "Run actions (single backtest / walk-forward)",
        "secondary": "Downloads / open dialogs",
        "ghost": "Inline controls (toggle series, expand)"
      },
      "micro_interactions": {
        "hover": "Increase border alpha + subtle background lift (no scale on dense tables)",
        "press": "active:translate-y-[1px] active:shadow-none",
        "focus": "ring-2 ring-[hsl(var(--ring))] ring-offset-0"
      }
    },
    "forms": {
      "run_panel": {
        "instrument_select": "Select",
        "tau1_tau2": "Slider + Input (dual entry) with clamped ranges",
        "N": "Slider + Input",
        "presets": "Menubar or DropdownMenu for saved parameter sets",
        "validation": "Inline helper text in muted-foreground; errors in destructive"
      }
    }
  },

  "charts_recharts": {
    "global_chart_style": {
      "font": "Use mono for axes ticks and tooltips",
      "grid": "CartesianGrid stroke=hsla(var(--hairline)/0.35) strokeDasharray='3 3'",
      "tooltip": "Custom tooltip with Card-like surface; show exact values with fixed decimals",
      "colors": {
        "equity": "hsl(var(--info))",
        "drawdown": "hsla(var(--neg)/0.55)",
        "monthly_pos": "hsl(var(--pos))",
        "monthly_neg": "hsl(var(--neg))",
        "regime_normal": "hsla(215,12%,70%,0.55)",
        "regime_toxic_cont": "hsla(var(--pos)/0.55)",
        "regime_toxic_rev": "hsla(var(--warn)/0.55)"
      }
    },
    "required_charts": {
      "EquityCurve": "LineChart (monotone) + reference line at 0",
      "DrawdownUnderwater": "AreaChart (negative area) with gradient fill kept subtle",
      "MonthlyReturns": "BarChart with conditional fill by sign",
      "PnLHistogram": "BarChart with bins; highlight mean/median lines",
      "HMMPosteriors": "AreaChart stacked (3 states)",
      "VolRegimeTimeline": "ComposedChart or custom banded background + line",
      "ParamSensitivityHeatmap": "Use a custom SVG heatmap grid (div grid) or recharts ScatterChart with square points; include legend"
    },
    "empty_states": {
      "before_run": "Show skeleton chart frames + a centered hint: 'Run a backtest to populate charts'",
      "no_trades": "Show neutral empty state with explanation (not an error)"
    }
  },

  "terminal_console": {
    "visual_spec": {
      "container": "Card with inset border; background slightly darker than cards (surface-0)",
      "text": "font-mono text-xs leading-5",
      "lines": "timestamp + stage + message; stage colored info/warn/neg",
      "scroll": "ScrollArea with auto-scroll toggle"
    },
    "interaction": {
      "polling": "Show last polled time; if stale >10s show warn chip",
      "copy": "Copy-to-clipboard button for selected lines",
      "download": "Download log as .txt"
    }
  },

  "tables": {
    "trade_log_density": {
      "row_height": "h-9 md:h-10",
      "cell_padding": "px-2 py-1",
      "header": "sticky top-0 bg-[hsl(var(--card))] backdrop-blur supports-[backdrop-filter]:bg-[hsl(var(--card)/0.85)]",
      "zebra": "odd:bg-[hsl(var(--muted)/0.35)]",
      "hover": "hover:bg-[hsl(var(--muted)/0.55)]",
      "numeric_alignment": "text-right font-mono tabular-nums",
      "direction_badge": "LONG=pos tint, SHORT=neg tint"
    },
    "sorting_filtering": {
      "controls": "Use Input for search, Select for direction, date range optional later",
      "csv_download": "Button variant=secondary with download icon"
    }
  },

  "motion": {
    "principles": [
      "Motion is functional: indicate state changes, not decoration.",
      "Prefer opacity/height transitions; avoid scaling dense tables.",
      "Respect prefers-reduced-motion."
    ],
    "framer_motion_usage": {
      "tab_panels": "AnimatePresence with fade+slide (y: 6px) 160–220ms",
      "kpi_updates": "Number tick animation optional; otherwise flash highlight (bg tint) for 300ms",
      "job_running": "Subtle animated scanline in console header + spinner"
    }
  },

  "accessibility": {
    "contrast": "All text must meet WCAG AA on dark surfaces; avoid low-contrast gray-on-gray.",
    "focus": "Visible focus ring on all interactive elements (ring token).",
    "keyboard": "Tabs, Select, Sliders, Table sorting must be keyboard accessible.",
    "color_meaning": "Never rely on color alone: include +/− signs, PASS/FAIL text, and icons where helpful."
  },

  "data_testid_convention": {
    "rule": "All interactive and key informational elements MUST include data-testid (kebab-case, role-based).",
    "examples": [
      "data-testid=\"run-panel-instrument-select\"",
      "data-testid=\"run-panel-tau1-slider\"",
      "data-testid=\"run-panel-run-single-button\"",
      "data-testid=\"run-panel-run-wfo-button\"",
      "data-testid=\"progress-console-container\"",
      "data-testid=\"acceptance-scorecard-verdict-chip\"",
      "data-testid=\"metrics-grid-sharpe-value\"",
      "data-testid=\"trade-log-search-input\"",
      "data-testid=\"trade-log-download-csv-button\"",
      "data-testid=\"chart-equity-curve-container\""
    ]
  },

  "image_urls": [
    {
      "category": "background-texture",
      "description": "Optional subtle hero/background image for the app shell (use as very low-opacity overlay, not a section hero).",
      "url": "https://images.pexels.com/photos/26954240/pexels-photo-26954240.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940"
    },
    {
      "category": "about-page-visual",
      "description": "Optional about/strategy page header visual (futuristic analytics interface).",
      "url": "https://images.pexels.com/photos/27141316/pexels-photo-27141316.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940"
    },
    {
      "category": "ambient-overlay",
      "description": "Optional ambient overlay image (use opacity <= 0.04) to add terminal vibe without harming readability.",
      "url": "https://images.unsplash.com/photo-1643962578277-0e7e2f7b7c63?crop=entropy&cs=srgb&fm=jpg&ixlib=rb-4.1.0&q=85"
    }
  ],

  "implementation_notes_react_js": {
    "file_conventions": "Project uses .js (not .tsx). Keep components in JS, use PropTypes only if already used; otherwise rely on runtime checks + clear naming.",
    "tailwind_tokens": {
      "app_shell": "min-h-screen bg-[hsl(var(--background))] text-[hsl(var(--foreground))]",
      "card": "bg-[hsl(var(--card))] border border-[hsl(var(--border))]",
      "hairline": "border-[hsl(var(--hairline))]",
      "mono": "font-mono tabular-nums",
      "kpi_label": "text-xs uppercase tracking-wide text-muted-foreground",
      "kpi_value": "text-2xl font-semibold font-mono tabular-nums"
    },
    "suggested_app_css_cleanup": [
      "Remove default CRA App.css centering patterns; do not use .App { text-align:center }.",
      "Set body background via tokens; keep App.css minimal or delete if unused."
    ],
    "polling_ux": {
      "wfo_job": "When WFO running: disable run buttons, show Progress bar + status pill + console streaming; allow cancel if backend supports.",
      "non_blocking": "Main UI remains navigable; charts show last completed snapshot with 'Updating…' chip."
    }
  },

  "instructions_to_main_agent": [
    "Update /app/frontend/src/index.css tokens: treat this as dark-first; set :root to the dark palette (or force .dark on html).",
    "Apply IBM Plex Sans + IBM Plex Mono via Google Fonts in index.html or CSS import; set --font-sans/--font-mono and use font-mono tabular-nums for all numeric fields.",
    "Build the dashboard as a left rail + top bar + Tabs main area. Use shadcn Card for every panel; keep headers compact.",
    "Implement Acceptance Scorecard with Badge chips (PASS/FAIL) and an overall verdict chip; failure must look neutral and intentional.",
    "Implement Live Progress Console as a terminal-like ScrollArea with mono text; include auto-scroll toggle and last-updated timestamp.",
    "Use recharts with custom tooltip and mono ticks; keep gridlines subtle; use semantic colors for positive/negative.",
    "Trade Log table must be dense, sortable, filterable, with sticky header and CSV download.",
    "Add data-testid to every interactive control and key metric value element (kebab-case).",
    "Avoid gradients except tiny ambient overlays (<20% viewport). No purple/pink gradients."
  ],

  "general_ui_ux_design_guidelines": [
    "You must **not** apply universal transition. Eg: `transition: all`. This results in breaking transforms. Always add transitions for specific interactive elements like button, input excluding transforms",
    "You must **not** center align the app container, ie do not add `.App { text-align: center; }` in the css file. This disrupts the human natural reading flow of text",
    "NEVER: use AI assistant Emoji characters like`🤖🧠💭💡🔮🎯📚🎭🎬🎪🎉🎊🎁🎀🎂🍰🎈🎨🎰💰💵💳🏦💎🪙💸🤑📊📈📉💹🔢🏆🥇 etc for icons. Always use **FontAwesome cdn** or **lucid-react** library already installed in the package.json",
    "\n **GRADIENT RESTRICTION RULE**\nNEVER use dark/saturated gradient combos (e.g., purple/pink) on any UI element.  Prohibited gradients: blue-500 to purple 600, purple 500 to pink-500, green-500 to blue-500, red to pink etc\nNEVER use dark gradients for logo, testimonial, footer etc\nNEVER let gradients cover more than 20% of the viewport.\nNEVER apply gradients to text-heavy content or reading areas.\nNEVER use gradients on small UI elements (<100px width).\nNEVER stack multiple gradient layers in the same viewport.\n\n**ENFORCEMENT RULE:**\n    • Id gradient area exceeds 20% of viewport OR affects readability, **THEN** use solid colors\n\n**How and where to use:**\n   • Section backgrounds (not content backgrounds)\n   • Hero section header content. Eg: dark to light to dark color\n   • Decorative overlays and accent elements only\n   • Hero section with 2-3 mild color\n   • Gradients creation can be done for any angle say horizontal, vertical or diagonal\n\n- For AI chat, voice application, **do not use purple color. Use color like light green, ocean blue, peach orange etc",
    "\n- Every interaction needs micro-animations - hover states, transitions, parallax effects, and entrance animations. Static = dead.",
    "\n- Use 2-3x more spacing than feels comfortable. Cramped designs look cheap.",
    "\n- Subtle grain textures, noise overlays, custom cursors, selection states, and loading animations: separates good from extraordinary.",
    "\n- Before generating UI, infer the visual style from the problem statement (palette, contrast, mood, motion) and immediately instantiate it by setting global design tokens (primary, secondary/accent, background, foreground, ring, state colors), rather than relying on any library defaults. Don't make the background dark as a default step, always understand problem first and define colors accordingly\n    Eg: - if it implies playful/energetic, choose a colorful scheme\n           - if it implies monochrome/minimal, choose a black–white/neutral scheme\n\n**Component Reuse:**\n\t- Prioritize using pre-existing components from src/components/ui when applicable\n\t- Create new components that match the style and conventions of existing components when needed\n\t- Examine existing components to understand the project's component patterns before creating new ones\n\n**IMPORTANT**: Do not use HTML based component like dropdown, calendar, toast etc. You **MUST** always use `/app/frontend/src/components/ui/ ` only as a primary components as these are modern and stylish component\n\n**Best Practices:**\n\t- Use Shadcn/UI as the primary component library for consistency and accessibility\n\t- Import path: ./components/[component-name]\n\n**Export Conventions:**\n\t- Components MUST use named exports (export const ComponentName = ...)\n\t- Pages MUST use default exports (export default function PageName() {...})\n\n**Toasts:**\n  - Use `sonner` for toasts\"\n  - Sonner component are located in `/app/src/components/ui/sonner.tsx`\n\nUse 2–4 color gradients, subtle textures/noise overlays, or CSS-based noise to avoid flat visuals."
  ]
}
