#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int main(int argc, char* argv[])
{
    int sockfd, portno, n;    
    struct sockaddr_in serv_addr;
    struct hostent *server;

    char buffer[256];
    if (argc < 3)
    {
        fprintf(stderr, "usage %s hostname portno\n", argv[0]);
        exit(0);
    }
    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        fprintf(stderr, "ERROR when creating socket\n");
        exit(1);
    }
    server = gethostbyname(argv[1]);
    if (server == NULL)
    {
        fprintf(stderr, "ERROR, no such host\n");
        exit(0);
    }
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    memcpy((char *) &serv_addr.sin_addr.s_addr,
           (char *) server->h_addr, server->h_length);
    serv_addr.sin_port = htons(portno);

    if (connect(sockfd, 
                (struct sockaddr *) &serv_addr,
                sizeof(serv_addr)) < 0)
    {
        fprintf(stderr, "ERROR connecting to host");
        exit(1);
    }
    printf("Please enter the message you want to send: ");
    fgets(buffer, 255, stdin);
    n = write(sockfd, buffer, strlen(buffer));
    if (n < 0)
    {
        fprintf(stderr, "ERROR writing to socket");
        exit(1);
    }
    memset(buffer, 0, 256);
    n = read(sockfd, buffer, 255);
    if (n < 0)
    {
        fprintf(stderr, "ERROR reading from socket");
        exit(1);
    }
    printf("%s\n", buffer);
    close(sockfd);
    return 0;
}
