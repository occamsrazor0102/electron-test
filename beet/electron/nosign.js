// No-op signing function for cross-platform builds without code signing cert
exports.default = async function(configuration) {
  // Skip signing
};
