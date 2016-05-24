use std::io::{Read, Write};
use std::thread;
use std::sync::{mpsc, RwLock, Arc};
use std::net::SocketAddr;
use std::boxed::Box;

#[allow(unused_imports)]
use hyper::{Get, Post, StatusCode, RequestUri, Decoder, Encoder, Next, Control};
use hyper::header::ContentLength;
use hyper::net::HttpStream;
use hyper::server::{Server, Handler, Request, Response};

use server::{self, TaskModel};
use features;
use metrics;
