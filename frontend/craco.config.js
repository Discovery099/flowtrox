// craco.config.js
const path = require("path");
require("dotenv").config();

// Check if we're in development/preview mode (not production build)
// Craco sets NODE_ENV=development for start, NODE_ENV=production for build
const isDevServer = process.env.NODE_ENV !== "production";

// Environment variable overrides
const config = {
  enableHealthCheck: process.env.ENABLE_HEALTH_CHECK === "true",
};

// Conditionally load health check modules only if enabled
let WebpackHealthPlugin;
let setupHealthEndpoints;
let healthPluginInstance;

if (config.enableHealthCheck) {
  WebpackHealthPlugin = require("./plugins/health-check/webpack-health-plugin");
  setupHealthEndpoints = require("./plugins/health-check/health-endpoints");
  healthPluginInstance = new WebpackHealthPlugin();
}

let webpackConfig = {
  eslint: {
    configure: {
      extends: ["plugin:react-hooks/recommended"],
      rules: {
        "react-hooks/rules-of-hooks": "error",
        "react-hooks/exhaustive-deps": "warn",
      },
    },
  },
  webpack: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
    configure: (webpackConfig) => {

      // Add ignored patterns to reduce watched directories
        webpackConfig.watchOptions = {
          ...webpackConfig.watchOptions,
          ignored: [
            '**/node_modules/**',
            '**/.git/**',
            '**/build/**',
            '**/dist/**',
            '**/coverage/**',
            '**/public/**',
        ],
      };

      // Add health check plugin to webpack if enabled
      if (config.enableHealthCheck && healthPluginInstance) {
        webpackConfig.plugins.push(healthPluginInstance);
      }
      return webpackConfig;
    },
  },
};

webpackConfig.devServer = (devServerConfig) => {
  // ---------------------------------------------------------------------
  // webpack-dev-server v4 -> v5 compatibility shim.
  // CRA 5 (react-scripts) injects `onBeforeSetupMiddleware` and
  // `onAfterSetupMiddleware` (v4 API). The project pins webpack-dev-server v5
  // (via resolutions), whose schema rejects those keys and crashes the dev
  // server. We migrate them to v5's `setupMiddlewares`.
  const before = devServerConfig.onBeforeSetupMiddleware;
  const after = devServerConfig.onAfterSetupMiddleware;
  delete devServerConfig.onBeforeSetupMiddleware;
  delete devServerConfig.onAfterSetupMiddleware;

  const prevSetupMiddlewares = devServerConfig.setupMiddlewares;

  devServerConfig.setupMiddlewares = (middlewares, devServer) => {
    if (typeof before === "function") before(devServer);
    if (typeof prevSetupMiddlewares === "function") {
      middlewares = prevSetupMiddlewares(middlewares, devServer);
    }
    if (typeof after === "function") after(devServer);

    // Add health check endpoints if enabled
    if (config.enableHealthCheck && setupHealthEndpoints && healthPluginInstance) {
      setupHealthEndpoints(devServer, healthPluginInstance);
    }

    return middlewares;
  };

  // Allow the preview/proxy host through wds v5 host checking.
  devServerConfig.allowedHosts = "all";

  // v4 `https` option was replaced by `server` in v5.
  if (Object.prototype.hasOwnProperty.call(devServerConfig, "https")) {
    const https = devServerConfig.https;
    delete devServerConfig.https;
    if (https && typeof https === "object") {
      devServerConfig.server = { type: "https", options: https };
    } else {
      devServerConfig.server = https ? "https" : "http";
    }
  }

  return devServerConfig;
};

// Wrap with visual edits (automatically adds babel plugin, dev server, and overlay in dev mode)
// NOTE: Gated off by default because the visual-edits dev-server wrapper injects
// `onAfterSetupMiddleware` (webpack-dev-server v4 API) which is incompatible with the
// installed webpack-dev-server v5 and crashes the dev server. Set ENABLE_VISUAL_EDITS=true
// to re-enable once a compatible visual-edits version is available.
if (isDevServer && process.env.ENABLE_VISUAL_EDITS === "true") {
  try {
    const { withVisualEdits } = require("@emergentbase/visual-edits/craco");
    webpackConfig = withVisualEdits(webpackConfig);
  } catch (err) {
    if (err.code === 'MODULE_NOT_FOUND' && err.message.includes('@emergentbase/visual-edits/craco')) {
      console.warn(
        "[visual-edits] @emergentbase/visual-edits not installed — visual editing disabled."
      );
    } else {
      throw err;
    }
  }
}

module.exports = webpackConfig;
