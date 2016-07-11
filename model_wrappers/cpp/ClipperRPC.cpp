#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include "ClipperRPC.h"
#include "Model.h"
#define SHUTDOWN_CODE   0
#define FIXEDINT_CODE   1
#define FIXEDFLOAT_CODE 2
#define FIXEDBYTE_CODE  3
#define VARINT_CODE     4
#define VARFLOAT_CODE   5
#define VARBYTE_CODE    6
#define STRING_CODE     7

void error(const char* error_msg) {
    fprintf(stderr, "%s\n", error_msg);
    exit(1);
}

bool is_fixed_format(char input_type) {
    return input_type == FIXEDINT_CODE ||
           input_type == FIXEDFLOAT_CODE || input_type == FIXEDBYTE_CODE;
}

bool is_var_format(char input_type) {
    return input_type == VARINT_CODE ||
           input_type == VARFLOAT_CODE || input_type == VARBYTE_CODE;
}

void ClipperRPC::handle(int newsockfd) {
    char buffer[256], *data;
    int i, j, n, bytes_read, header_bytes = 5;
    int additional_header_bytes, total_bytes_expected;
    uint32_t input_type, num_inputs, input_len, shutdown_msg;
    vector<vector<char> > v_byte;
    vector<vector<double> > v_double;
    vector<vector<uint32_t> > v_int;
    vector<vector<string> > v_str;
    vector<double> *predictions;

    printf("Handling new connection\n");
    while (true) {
        bytes_read = 0;
        while (bytes_read < header_bytes) {
            n = read(newsockfd, &buffer[bytes_read], header_bytes);
            if (n < 0) {
                error("ERROR reading from socket");
            }
            bytes_read += n;
        }

        input_type = buffer[0];
        memcpy(&num_inputs, &buffer[1], 4);
        if (input_type == SHUTDOWN_CODE) {
            printf("Shutting down connection\n");
            shutdown_msg = 1234;
            write(newsockfd, &shutdown_msg, 4);
            return;
        }
        if (is_fixed_format(input_type)) {
            additional_header_bytes = 4;
            while (bytes_read < header_bytes + additional_header_bytes) {
                n = read(newsockfd, &buffer[bytes_read], header_bytes);
                if (n < 0) {
                    error("ERROR reading from socket");
                }
                bytes_read += n;
            }
            memcpy(&input_len, &buffer[header_bytes], additional_header_bytes);
            bytes_read -= (header_bytes + additional_header_bytes);
            if (input_type == FIXEDBYTE_CODE) {
                total_bytes_expected = input_len*num_inputs;
                data = new char[total_bytes_expected];
                memset(data, 0, total_bytes_expected);
                memcpy(data, &buffer[header_bytes + additional_header_bytes],
                       bytes_read);
                while (bytes_read < total_bytes_expected) {
                    n = read(newsockfd, &data[bytes_read], 256);
                    if (n < 0) {
                        error("ERROR reading from socket");
                    }
                    bytes_read += n;
                }
                v_byte = vector<vector<char> >(num_inputs,
                                               vector<char>(input_len, 0));
                for (i = 0; i < num_inputs; i++) {
                    for (j = 0; j < input_len; j++) {
                        v_byte[i][j] = data[i*input_len + j];
                    }
                }
            } else if (input_type == FIXEDFLOAT_CODE) {
                total_bytes_expected = 8*input_len*num_inputs;
                data = new char[total_bytes_expected];
                memset(data, 0, total_bytes_expected);
                memcpy(data, &buffer[header_bytes + additional_header_bytes],
                       bytes_read);
                while (bytes_read < total_bytes_expected) {
                    n = read(newsockfd, &data[bytes_read], 256);
                    if (n < 0) {
                        error("ERROR reading from socket");
                    }
                    bytes_read += n;
                }
                v_double = vector<vector<double> >(num_inputs,
                                                   vector<double>(input_len, 0));
                for (i = 0; i < num_inputs; i++) {
                    for (j = 0; j < input_len; j++) {
                        v_double[i][j] = *((double *) &data[8*(i*input_len + j)]);
                    }
                }
            } else if (input_type == FIXEDINT_CODE) {
                total_bytes_expected = 4*input_len*num_inputs;
                data = new char[total_bytes_expected];
                memset(data, 0, total_bytes_expected);
                memcpy(data, &buffer[header_bytes + additional_header_bytes],
                       bytes_read);
                while (bytes_read < total_bytes_expected) {
                    n = read(newsockfd, &data[bytes_read], 256);
                    if (n < 0) {
                        error("ERROR reading from socket");
                    }
                    bytes_read += n;
                }
                v_int = vector<vector<uint32_t> >(num_inputs,
                                                  vector<uint32_t>(input_len, 0));
                for (i = 0; i < num_inputs; i++) {
                    for (j = 0; j < input_len; j++) {
                        v_int[i][j] = *((uint32_t *) &data[4*(i*input_len + j)]);
                    }
                }
            }
        } else if (is_var_format(input_type)) {
            error("ERROR: variable lengths not yet implemented");
        } else if (input_type == STRING_CODE) {
            error("ERROR: String type not yet implemented");
        } else {
            error("ERROR: Invalid input type");
        }

        if (input_type == FIXEDBYTE_CODE || input_type == VARBYTE_CODE) {
            predictions = model->predict_bytes(v_byte);
        } else if (input_type == FIXEDFLOAT_CODE || input_type == VARFLOAT_CODE) {
            predictions = model->predict_floats(v_double);
        } else if (input_type == FIXEDINT_CODE || input_type == VARINT_CODE) {
            predictions = model->predict_ints(v_int);
        } else {
            predictions = model->predict_strings(v_str);
        }
        for (int i = 0; i < predictions->size(); i++) {
            write(newsockfd, &(*predictions)[i], 8);
            printf("Prediction %d: %f\n", i, (*predictions)[i]);
        }

        delete predictions;
        free(data);
    }
}

void ClipperRPC::serve_forever() {
    int sockfd, newsockfd, pid;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        error("ERROR opening socket");
    }
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    inet_aton(s_addr, &serv_addr.sin_addr);
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        error("ERROR binding socket");
    }
    listen(sockfd, 10);
    clilen = sizeof(cli_addr);
    while (true) {
        newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
        if (newsockfd < 0) {
            error("ERROR opening client connection");
        }
        handle(newsockfd);
        close(newsockfd);
        printf("Closing connection\n");
    }
}

ClipperRPC::ClipperRPC(std::unique_ptr<Model> &model,
                       char *s_addr, int portno) :
    model(std::move(model)), s_addr(s_addr), portno(portno) {}

