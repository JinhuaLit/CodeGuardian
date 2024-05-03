# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import pickle
from lime.lime_text import LimeTextExplainer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("lebretou/code-human-ai")
model = AutoModelForSequenceClassification.from_pretrained("lebretou/code-human-ai")


with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Check the device used by the model
device = model.device

human_code_1 = '''
"/*
 * Simple HTTP Proxy in C
 *
 * A simple HTTP proxy that listens on a user-defined port, accepts HTTP
 * requests from a client, forwards them to a server, accepts the server's
 * response and forwards it back to the client.
 *
 * Written by John Smith.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

/* Constants */
#define DEFAULT_PORT 8080
#define MAX_REQUEST_SIZE 8192
#define MAX_RESPONSE_SIZE 65536

/* Function declarations */
int create_listen_socket(int port);
int process_client_request(int client_fd, const char* server_name, int server_port);
int send_request_to_server(int server_fd, const char* buffer, int length);
int read_response_from_server(int server_fd, char* buffer, int max_size);

/* Signal handler */
void handle_sigint(int sig)
{
    printf(""Caught SIGINT, shutting down...\n"");
    exit(0);
}

/* Main function */
int main(int argc, char** argv)
{
    int listen_fd, client_fd, server_fd, port;
    struct sockaddr_storage client_addr;
    socklen_t client_len = sizeof(client_addr);
    char client_ip[INET6_ADDRSTRLEN];
    char request_buffer[MAX_REQUEST_SIZE], response_buffer[MAX_RESPONSE_SIZE];
    int bytes_read;
    pid_t pid;
    struct hostent* server_info;

    /* Register signal handler */
    signal(SIGINT, handle_sigint);

    /* Parse command-line arguments */
    if (argc < 2) {
        port = DEFAULT_PORT;
    } else {
        port = atoi(argv[1]);
    }

    /* Create listening socket */
    listen_fd = create_listen_socket(port);
    printf(""Listening on port hahhahah caonima %d...\n"", port);

    /* Main loop */
    while (true) {
        /* Accept incoming connections */
        client_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd == -1) {
            perror(""accept"");
            continue;
        }

        /* Fork a new process to handle the request */
        pid = fork();
        if (pid == -1) {
            perror(""fork"");
            close(client_fd);
            continue;
        } else if (pid == 0) {
            /* Child process */
            close(listen_fd);

            /* Get client IP address */
            if (client_addr.ss_family == AF_INET) {
                inet_ntop(AF_INET, &(((struct sockaddr_in*)&client_addr)->sin_addr),
                        client_ip, INET_ADDRSTRLEN);
            } else {
                inet_ntop(AF_INET6, &(((struct sockaddr_in6*)&client_addr)->sin6_addr),
                        client_ip, INET6_ADDRSTRLEN);
            }

            /* Read client request */
            bytes_read = read(client_fd, request_buffer, MAX_REQUEST_SIZE);
            if (bytes_read == -1) {
                perror(""read"");
                close(client_fd);
                exit(1);
            }

            /* Process client request */
            server_info = gethostbyname(""www.google.com"");
            server_fd = process_client_request(client_fd, server_info->h_name, 80);

            /* Send request to server */
            if (send_request_to_server(server_fd, request_buffer, bytes_read) == -1) {
                perror(""send"");
                close(server_fd);
                close(client_fd);
                exit(1);
            }

            /* Read response from server */
            bytes_read = read_response_from_server(server_fd, response_buffer, MAX_RESPONSE_SIZE);
            if (bytes_read == -1) {
                perror(""read"");
                close(server_fd);
                close(client_fd);
                exit(1);
            }

            /* Send response to client */
            if (write(client_fd, response_buffer, bytes_read) == -1) {
                perror(""write"");
                close(server_fd);
                close(client_fd);
                exit(1);
            }

            /* Close sockets and exit */
            close(server_fd);
            close(client_fd);
            exit(0);
        } else {
            /* Parent process */
            close(client_fd);
        }
    }

    return 0;

}"
'''



# Tokenize the input text and ensure it's on the right device
encoding = tokenizer(human_code_1, truncation=True, padding=True, return_tensors="pt")
encoding = {k: v.to(device) for k, v in encoding.items()}  # Move the encoding to the same device as the model

# Predict
with torch.no_grad():
    outputs = model(**encoding, output_attentions=True)
    predictions = outputs.logits.argmax(-1)

# Convert the numerical prediction back to a label
predicted_label = label_encoder.inverse_transform(predictions.cpu().numpy())[0]  # Move predictions to CPU for numpy conversion

print(f"Predicted label: {predicted_label}")


# Attention weights from the last layer
attention_weights = outputs.attentions[-1].squeeze().cpu().numpy()

# Tokenized input ids
input_ids = encoding['input_ids'].cpu().numpy()[0]

# Code split into lines
lines = human_code_1.split('\n')

# Initialize variables for attention processing
line_attention_weights = []
current_line_weight = 0
line_idx = 0

# Compute attention weight per line
for token_id, weight in zip(input_ids, attention_weights[0]):
    if token_id == tokenizer.eos_token_id:
        break

    token = tokenizer.decode([token_id])
    
    if token == tokenizer.pad_token:
        continue

    if token == '\n':
        line_attention_weights.append((lines[line_idx], current_line_weight))
        line_idx += 1
        current_line_weight = 0
    else:
        current_line_weight += weight.sum()  # Ensure to sum if weight is an array

# Handle the last line if needed
if current_line_weight > 0:
    line_attention_weights.append((lines[line_idx], current_line_weight))

# Normalize weights
total_weight = sum(weight for _, weight in line_attention_weights)
line_attention_weights = [(line, weight / total_weight) for line, weight in line_attention_weights]

# Sort lines by weight and print the top 5
top_lines = sorted(line_attention_weights, key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Lines by Attention Weight:")
for line, weight in top_lines:
    print(f"{line.strip()}: {weight}")