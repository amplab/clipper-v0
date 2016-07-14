#ifndef __CLIPPERRPC_H__
#define __CLIPPERRPC_H__
#include "Model.h"

class ClipperRPC {
    public:
        ClipperRPC(std::unique_ptr<Model> &model, char *s_addr, int portno);

        std::unique_ptr<Model> model;

        char* s_addr;

        int portno;

        void handle(int newsockfd);

        void serve_forever(void);
};

#endif
