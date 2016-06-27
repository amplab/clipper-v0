#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#define FIXEDINT_CODE   1
#define FIXEDFLOAT_CODE 2
#define FIXEDBYTE_CODE  3
#define VARINT_CODE     4
#define VARFLOAT_CODE   5
#define VARBYTE_CODE    6
#define STRING_CODE     7

void error(const char* error_msg)
{
    fprintf(stderr, "%s\n", error_msg);
    exit(1);
}

void handle_connection(int newsockfd)
{
    int n;
    char buffer[256];

    memset(buffer, 0, 256);
    n = read(newsockfd, buffer, 255);
    if (n < 0)
        error("ERROR reading from socket");
    printf("Here is the message: %s\n", buffer);
    n = write(newsockfd, "I got your message", 18);
    if (n < 0)
        error("ERROR writing to socket");
}

bool is_fixed_format(char input_type)
{
    return input_type == FIXEDINT_CODE ||
           input_type == FIXEDFLOAT_CODE || input_type == FIXEDBYTE_CODE;
}

bool is_var_format(char input_type)
{
    return input_type == VARINT_CODE ||
           input_type == VARFLOAT_CODE || input_type == VARBYTE_CODE;
}

void handle_rpc_connection(int newsockfd)
{
    char buffer[256], *data;
    int n, bytes_read, header_bytes = 5;
    int additional_header_bytes, total_bytes_expected;
    uint32_t input_type, num_inputs, input_len;

    bytes_read = 0;
    while (bytes_read < header_bytes)
    {
        n = read(newsockfd, &buffer[bytes_read], header_bytes);
        if (n < 0)
            error("ERROR reading from socket");
        bytes_read += n;
    }

    input_type = buffer[0];
    memcpy(&num_inputs, &buffer[1], 4);
    num_inputs = ntohl(num_inputs);
    if (is_fixed_format(input_type))
    {
        // printf("Received vals: %d %d", input_type, num_inputs);
        additional_header_bytes = 4;
        while (bytes_read < header_bytes + additional_header_bytes)
        {
            n = read(newsockfd, &buffer[bytes_read], header_bytes);
            if (n < 0)
                error("ERROR reading from socket");
            bytes_read += n;
        }
        memcpy(&input_len, &buffer[header_bytes], additional_header_bytes);
        input_len = ntohl(input_len);
        bytes_read -= (header_bytes + additional_header_bytes);
        if (input_type == FIXEDBYTE_CODE)
            total_bytes_expected = input_len*num_inputs;
        else if (input_type == FIXEDFLOAT_CODE)
            total_bytes_expected = 8*input_len*num_inputs;
        else if (input_type == FIXEDINT_CODE)
            total_bytes_expected = 4*input_len*num_inputs;
        data = new char[total_bytes_expected];
        memcpy(data, &buffer[header_bytes + additional_header_bytes],
               bytes_read);
        while (bytes_read < total_bytes_expected)
        {
            n = read(newsockfd, &data[bytes_read], 256);
            if (n < 0)
                error("ERROR reading from socket");
            bytes_read += n;
        }

        /*
        if (input_type == FIXEDBYTE_CODE)
        {
            total_bytes_expected = input_len*num_inputs;
            data = new char[total_bytes_expected];
            while (bytes_read < total_bytes_expected)
            {
                n = read(newsockfd, &data[bytes_read], 256);
                if (n < 0)
                    error("ERROR reading from socket");
                bytes_read += n;
            }

        }
        else if (input_type == FIXEDFLOAT_CODE)
        {
            total_bytes_expected = 8*input_len*num_inputs;
            data = new char[total_bytes_expected];

        }
        else if (input_type == FIXEDINT_CODE)
        {
            total_bytes_expected = 4*input_len*num_inputs;
            data = new char[total_bytes_expected];

        } */
    }
    else if (is_var_format(input_type))
        error("ERROR: variable lengths not yet implemented");
    else if (input_type == STRING_CODE)
        error("ERROR: String type not yet implemented");
    else
        error("ERROR: Invalid input type");
}

int main(int argc, char* argv[])
{
    int sockfd, newsockfd, portno, pid;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    if (argc < 2)
        error("ERROR, no port provided");
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("ERROR opening socket");
    memset(&serv_addr, 0, sizeof(serv_addr));
    portno = atoi(argv[1]);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr,
            sizeof(serv_addr)) < 0)
        error("ERROR binding socket");
    listen(sockfd, 10);
    clilen = sizeof(cli_addr);
    while (true) {
        newsockfd = accept(sockfd,
                    (struct sockaddr *) &cli_addr,
                    &clilen);
        if (newsockfd < 0)
            error("ERROR opening client connection");

        pid = fork();
        if (pid < 0)
            error("ERROR forking process");

        if (pid == 0)
        {
            close(sockfd);
            // handle_connection(newsockfd);
            handle_rpc_connection(newsockfd);
            exit(0);
        }
        else close(newsockfd);

    }
    return 0;
}
