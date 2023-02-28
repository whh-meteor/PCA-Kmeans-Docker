

// const http = require('http');
// const httpProxy = require('http-proxy');

// // 创建一个 HTTP 代理服务器
// const proxy = httpProxy.createProxyServer();

// // 创建一个 HTTP 服务器，用于监听请求并将它们转发到目标服务器
// const server = http.createServer((req, res) => {
//   // 将请求转发到 localhost:3000 或 localhost:3001
//   proxy.web(req, res, { target: 'http://localhost:5000' });
//  // proxy.web(req, res, { target: 'http://localhost:3001' });
// });

// // 启动服务器，监听 localhost:8000 端口
// server.listen(5000);
// console.log('Server running at http://127.0.0.1:5000/');
